/*
 * Copyright 2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * The code is developed based on the original word2vec implementation from Google:
 * https://code.google.com/archive/p/word2vec/
 */

#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <stdint.h>

#ifdef USE_MKL
//#include <cblas.h>
#include <armpl.h>
#endif

using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000

#define _mm_malloc(size, alignment) aligned_alloc(alignment, size)
#define _mm_free(ptr) free(ptr)

struct vocab_word {
    uint64_t cn;
    char *word;
};

class sequence {
public:
    size_t *indices;
    int64_t *meta;
    size_t length;

    sequence(size_t len) {
        length = len;
        indices = (size_t *) _mm_malloc(length * sizeof(size_t), 64);
        meta = (int64_t *) _mm_malloc(length * sizeof(int64_t), 64);
    }

    ~sequence() {
        _mm_free(indices);
        _mm_free(meta);
    }
};

int64_t binary = 0, debug_mode = 2;
bool disk = false;
size_t negative = 5, min_count = 5, num_threads = 12, min_reduce = 1, iter = 5, window = 5, batch_size = 11;
size_t vocab_max_size = 1000, vocab_size = 0, hidden_size = 100;
uint64_t train_words = 0, file_size = 0;
float_t alpha = 0.025f, sample = 1e-3f;
const float_t EXP_RESOLUTION = EXP_TABLE_SIZE / (MAX_EXP * 2.0f);

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
const size_t vocab_hash_size = 60000000;  // Maximum 60 * 0.7 = 42M words in the vocabulary
const size_t table_size = 1e8;

struct vocab_word *vocab = NULL;
size_t *vocab_hash = NULL;
size_t *table = NULL;
float_t *Wih = NULL, *Woh = NULL, *expTable = NULL;

void InitUnigramTable() {
    table = (size_t *) _mm_malloc(table_size * sizeof(size_t), 64);

    const float_t power = 0.75f;
    double train_words_pow = 0.;
    #pragma omp parallel for num_threads(num_threads) reduction(+: train_words_pow)
    for (size_t i = 0; i < vocab_size; i++) {
        train_words_pow += pow(vocab[i].cn, power);
    }

    size_t i = 0;
    float_t d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (size_t a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (float_t) table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size)
            i = vocab_size - 1;
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    size_t a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch <= ' ') {
            if (a > 0) {
                if (ch == '\n')
                    ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *) "</s>");
                return;
            } else
                continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1)
            a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
size_t GetWordHash(char *word) {
    uint32_t hash = 0;
    size_t len = strlen(word);
    for (size_t i = 0; i < len; i++)
        hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int64_t SearchVocab(char *word) {
    size_t hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1)
            return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word))
            return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int64_t ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin))
        return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int64_t AddWordToVocab(char *word) {
    size_t hash;
    vocab[vocab_size].word = strdup(word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int64_t VocabCompare(const void *a, const void *b) {
    struct vocab_word *aa = (struct vocab_word *) a;
    struct vocab_word *bb = (struct vocab_word *) b;
    return (aa->cn > bb->cn) - (aa->cn < bb->cn);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    memset(vocab_hash, -1, vocab_hash_size * sizeof(size_t));

    size_t size = vocab_size;
    train_words = 0;
    for (size_t i = 0; i < size; i++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[i].cn < min_count) && (i != 0)) {
            vocab_size--;
            free(vocab[i].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            size_t hash = GetWordHash(vocab[i].word);
            while (vocab_hash[hash] != -1)
                hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = i;
            train_words += vocab[i].cn;
        }
    }
    vocab = (struct vocab_word *) realloc(vocab, vocab_size * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    size_t count = 0;
    for (int64_t i = 0; i < vocab_size; i++) {
        if (vocab[i].cn > min_reduce) {
            vocab[count].cn = vocab[i].cn;
            vocab[count].word = vocab[i].word;
            count++;
        } else {
            free(vocab[i].word);
        }
    }
    vocab_size = count;
    memset(vocab_hash, -1, vocab_hash_size * sizeof(size_t));

    for (size_t i = 0; i < vocab_size; i++) {
        // Hash will be re-computed, as it is not actual
        size_t hash = GetWordHash(vocab[i].word);
        while (vocab_hash[hash] != -1)
            hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = i;
    }
    min_reduce++;
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];

    memset(vocab_hash, -1, vocab_hash_size * sizeof(size_t));

    FILE *fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }

    train_words = 0;
    vocab_size = 0;
    AddWordToVocab((char *) "</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        int64_t i = SearchVocab(word);
        if (i == -1) {
            int64_t a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else
            vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7)
            ReduceVocab();
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    FILE *fo = fopen(save_vocab_file, "wb");
    for (size_t i = 0; i < vocab_size; i++)
        fprintf(fo, "%s %d\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    memset(vocab_hash, -1, vocab_hash_size * sizeof(size_t));

    char c;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        int64_t i = AddWordToVocab(word);
        fscanf(fin, "%d%c", &vocab[i].cn, &c);
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fclose(fin);

    // get file size
    FILE *fin2 = fopen(train_file, "rb");
    if (fin2 == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin2, 0, SEEK_END);
    file_size = ftell(fin2);
    fclose(fin2);
}

void InitNet() {
    Wih = (float_t *) _mm_malloc(vocab_size * hidden_size * sizeof(float_t), 64);
    Woh = (float_t *) _mm_malloc(vocab_size * hidden_size * sizeof(float_t), 64);
    if (!Wih || !Woh) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
    for (size_t i = 0; i < vocab_size; i++) {
        memset(Wih + i * hidden_size, 0.f, hidden_size * sizeof(float_t));
        memset(Woh + i * hidden_size, 0.f, hidden_size * sizeof(float_t));
    }

    // initialization
    uint64_t next_random = 1;
    for (size_t i = 0; i < vocab_size * hidden_size; i++) {
        next_random = next_random * (uint64_t) 25214903917 + 11;
        Wih[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
    }
}

uint64_t loadStream(FILE *fin, int64_t *stream, const uint64_t total_words) {
    uint64_t word_count = 0;
    while (!feof(fin) && word_count < total_words) {
        int64_t w = ReadWordIndex(fin);
        if (w == -1)
            continue;
        stream[word_count] = w;
        word_count++;
    }
    stream[word_count] = 0; // set the last word as "</s>"
    return word_count;
}

void Train_SGNS() {

#ifdef USE_MKL
#endif

    if (read_vocab_file[0] != 0) {
        ReadVocab();
    }
    else {
        LearnVocabFromTrainFile();
    }
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;

    InitNet();
    InitUnigramTable();

    float_t starting_alpha = alpha;
    uint64_t word_count_actual = 0;
    double start = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();
        size_t local_iter = iter;
        uint64_t  next_random = id;
        uint64_t word_count = 0, last_word_count = 0;
        size_t sentence_length = 0, sentence_position = 0;
        int64_t sen[MAX_SENTENCE_LENGTH] __attribute__((aligned(64)));

        // load stream
        FILE *fin = fopen(train_file, "rb");
        fseek(fin, file_size * id / num_threads, SEEK_SET);

        uint64_t local_train_words = train_words / num_threads + (train_words % num_threads > 0 ? 1 : 0);
        int64_t *stream;
        int64_t w;

        if (!disk) {
            stream = (int64_t *) _mm_malloc((local_train_words + 1) * sizeof(int64_t), 64);
            local_train_words = loadStream(fin, stream, local_train_words);
            fclose(fin);
        }

        // temporary memory
        float_t * inputM = (float_t *) _mm_malloc(batch_size * hidden_size * sizeof(float_t), 64);
        float_t * outputM = (float_t *) _mm_malloc((1 + negative) * hidden_size * sizeof(float_t), 64);
        float_t * outputMd = (float_t *) _mm_malloc((1 + negative) * hidden_size * sizeof(float_t), 64);
        float_t * corrM = (float_t *) _mm_malloc((1 + negative) * batch_size * sizeof(float_t), 64);

        int64_t inputs[2 * window + 1] __attribute__((aligned(64)));
        sequence outputs(1 + negative);

        #pragma omp barrier

        if (id == 0)
        {
            start = omp_get_wtime();
        }

        while (1) {
            if (word_count - last_word_count > 10000) {
                uint64_t diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                last_word_count = word_count;
                if (debug_mode > 1) {
                    double now = omp_get_wtime();
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk", 13, alpha,
                            word_count_actual / (float_t) (iter * train_words + 1) * 100,
                            word_count_actual / ((now - start) * 1000));
                    fflush(stdout);
                }
                alpha = starting_alpha * (1 - word_count_actual / (float_t) (iter * train_words + 1));
                if (alpha < starting_alpha * 0.0001f)
                    alpha = starting_alpha * 0.0001f;
            }
            if (sentence_length == 0) {
                while (1) {
                    if (disk) {
                        w = ReadWordIndex(fin);
                        if (feof(fin)) break;
                        if (w == -1) continue;
                    } else {
                        w = stream[word_count];
                    }
                    word_count++;
                    if (w == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        float_t ratio = (sample * train_words) / vocab[w].cn;
                        float_t ran = sqrtf(ratio) + ratio;
                        next_random = next_random * (uint64_t) 25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / 65536.f)
                            continue;
                    }
                    sen[sentence_length] = w;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                sentence_position = 0;
            }
            if ((disk && feof(fin)) || (word_count > local_train_words)) {
                uint64_t diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                local_iter--;
                if (local_iter == 0) break;
                word_count = 0;
                last_word_count = 0;
                sentence_length = 0;
                if (disk) {
                    fseek(fin, file_size * id / num_threads, SEEK_SET);
                }
                continue;
            }

            int64_t target = sen[sentence_position];
            outputs.indices[0] = target;
            outputs.meta[0] = 1;

            // get all input contexts around the target word
            next_random = next_random * (uint64_t) 25214903917 + 11;
            int64_t b = next_random % window;

            size_t num_inputs = 0;
            for (size_t i = b; i < 2 * window + 1 - b; i++) {
                if (i != window) {
                    size_t c = sentence_position - window + i;
                    if (c < 0)
                        continue;
                    if (c >= sentence_length)
                        break;
                    inputs[num_inputs] = sen[c];
                    num_inputs++;
                }
            }

            size_t num_batches = num_inputs / batch_size + ((num_inputs % batch_size > 0) ? 1 : 0);

            // start mini-batches
            for (size_t b = 0; b < num_batches; b++) {

                // generate negative samples for output layer
                size_t offset = 1;
                for (size_t k = 0; k < negative; k++) {
                    next_random = next_random * (uint64_t) 25214903917 + 11;
                    size_t sample = table[(next_random >> 16) % table_size];
                    if (!sample)
                        sample = next_random % (vocab_size - 1) + 1;
                    size_t* p = find(outputs.indices, outputs.indices + offset, sample);
                    if (p == outputs.indices + offset) {
                        outputs.indices[offset] = sample;
                        outputs.meta[offset] = 1;
                        offset++;
                    } else {
                        size_t idx = p - outputs.indices;
                        outputs.meta[idx]++;
                    }
                }
                outputs.meta[0] = 1;
                outputs.length = offset;

                // fetch input sub model
                size_t input_start = b * batch_size;
                size_t input_size  = min(batch_size, num_inputs - input_start);
                for (size_t i = 0; i < input_size; i++) {
                    memcpy(inputM + i * hidden_size, Wih + inputs[input_start + i] * hidden_size, hidden_size * sizeof(float_t));
                }
                // fetch output sub model
                size_t output_size = outputs.length;
                for (size_t i = 0; i < output_size; i++) {
                    memcpy(outputM + i * hidden_size, Woh + outputs.indices[i] * hidden_size, hidden_size * sizeof(float_t));
                }

#ifndef USE_MKL
                for (size_t i = 0; i < output_size; i++) {
                    size_t c = outputs.meta[i];
                    for (size_t j = 0; j < input_size; j++) {
                        float_t f = 0.f, g;
                        #pragma omp simd
                        for (size_t k = 0; k < hidden_size; k++) {
                            f += outputM[i * hidden_size + k] * inputM[j * hidden_size + k];
                        }
                        size_t label = (i ? 0 : 1);
                        if (f > MAX_EXP)
                            g = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            g = label * alpha;
                        else
                            g = (label - expTable[(size_t) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                        corrM[i * input_size + j] = g * c;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, output_size, input_size, hidden_size, 1.0f, outputM,
                        hidden_size, inputM, hidden_size, 0.0f, corrM, input_size);
                for (size_t i = 0; i < output_size; i++) {
                    int64_t c = outputs.meta[i];
                    size_t offset = i * input_size;
                    #pragma omp simd
                    for (size_t j = 0; j < input_size; j++) {
                        float_t f = corrM[offset + j];
                        size_t label = (i ? 0 : 1);
                        if (f > MAX_EXP)
                            f = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            f = label * alpha;
                        else
                            f = (label - expTable[(size_t) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                        corrM[offset + j] = f * c;
                    }
                }
#endif

#ifndef USE_MKL
                for (size_t i = 0; i < output_size; i++) {
                    for (size_t j = 0; j < hidden_size; j++) {
                        float_t f = 0.f;
                        #pragma omp simd
                        for (size_t k = 0; k < input_size; k++) {
                            f += corrM[i * input_size + k] * inputM[k * hidden_size + j];
                        }
                        outputMd[i * hidden_size + j] = f;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, hidden_size, input_size, 1.0f, corrM,
                        input_size, inputM, hidden_size, 0.0f, outputMd, hidden_size);
#endif

#ifndef USE_MKL
                for (size_t i = 0; i < input_size; i++) {
                    for (size_t j = 0; j < hidden_size; j++) {
                        float_t f = 0.f;
                        #pragma omp simd
                        for (size_t k = 0; k < output_size; k++) {
                            f += corrM[k * input_size + i] * outputM[k * hidden_size + j];
                        }
                        inputM[i * hidden_size + j] = f;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, input_size, hidden_size, output_size, 1.0f, corrM,
                        input_size, outputM, hidden_size, 0.0f, inputM, hidden_size);
#endif

                // subnet update
                for (size_t i = 0; i < input_size; i++) {
                    size_t src = i * hidden_size;
                    size_t des = inputs[input_start + i] * hidden_size;
                    #pragma omp simd
                    for (size_t j = 0; j < hidden_size; j++) {
                        Wih[des + j] += inputM[src + j];
                    }
                }

                for (size_t i = 0; i < output_size; i++) {
                    size_t src = i * hidden_size;
                    size_t des = outputs.indices[i] * hidden_size;
                    #pragma omp simd
                    for (size_t j = 0; j < hidden_size; j++) {
                        Woh[des + j] += outputMd[src + j];
                    }
                }

            }

            sentence_position++;
            if (sentence_position >= sentence_length) {
                sentence_length = 0;
            }
        }
        _mm_free(inputM);
        _mm_free(outputM);
        _mm_free(outputMd);
        _mm_free(corrM);
        if (disk) {
            fclose(fin);
        } else {
            _mm_free(stream);
        }
    }
}

int ArgPos(char *str, int argc, char **argv) {
    for (int a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            //if (a == argc - 1) {
            //    printf("Argument missing for %s\n", str);
            //    exit(1);
            //}
            return a;
        }
    return -1;
}

void saveModel() {
    // save the model
    FILE *fo = fopen(output_file, "wb");
    // Save the word vectors
    fprintf(fo, "%d %d\n", vocab_size, hidden_size);
    for (size_t a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (binary)
            for (size_t b = 0; b < hidden_size; b++)
                fwrite(&Wih[a * hidden_size + b], sizeof(float_t), 1, fo);
        else
            for (size_t b = 0; b < hidden_size; b++)
                fprintf(fo, "%f ", Wih[a * hidden_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

int main(int argc, char **argv) {
    if (argc == 1) {
        printf("parallel word2vec (sgns) in shared memory system\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-batch-size <int>\n");
        printf("\t\tThe batch size used for mini-batch training; default is 11 (i.e., 2 * window + 1)\n");
        printf("\t-disk\n");
        printf("\t\tStream text from disk during training, otherwise the text will be loaded into memory before training\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -binary 0 -iter 3\n\n");
        return 0;
    }

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;

    size_t i;
    if ((i = ArgPos((char *) "-size", argc, argv)) > 0)
        hidden_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-train", argc, argv)) > 0)
        strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)
        strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)
        strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)
        debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-binary", argc, argv)) > 0)
        binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
        alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-output", argc, argv)) > 0)
        strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
        window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
        sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
        negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-threads", argc, argv)) > 0)
        num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-iter", argc, argv)) > 0)
        iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
        min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-batch-size", argc, argv)) > 0)
        batch_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-disk", argc, argv)) > 0)
        disk = true;

    vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (size_t *) _mm_malloc(vocab_hash_size * sizeof(size_t), 64);
    expTable = (float_t *) _mm_malloc((EXP_TABLE_SIZE + 1) * sizeof(float_t), 64);
    for (i = 0; i < EXP_TABLE_SIZE + 1; i++) {
        expTable[i] = exp((i / (float_t) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                    // Precompute f(x) = x / (x + 1)
    }

    printf("number of threads: %d\n", num_threads);
    printf("number of iterations: %d\n", iter);
    printf("hidden size: %d\n", hidden_size);
    printf("number of negative samples: %d\n", negative);
    printf("window size: %d\n", window);
    printf("batch size: %d\n", batch_size);
    printf("starting learning rate: %.5f\n", alpha);
    printf("stream from disk: %d\n", disk);
    printf("starting training using file: %s\n\n", train_file);

    Train_SGNS();

    saveModel();
    return 0;
}
