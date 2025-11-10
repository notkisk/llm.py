#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_WORD_LEN 256
#define MAX_VOCAB_SIZE 100000
#define INITIAL_VOCAB_SIZE 256

static uint32_t hash_string(const char* str, size_t len) {
    uint32_t hash = 5381;
    for (size_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + (unsigned char)str[i];
    }
    return hash;
}

Tokenizer* tokenizer_new(void) {
    Tokenizer* tok = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (!tok) return NULL;
    
    tok->vocab = NULL;
    tok->vocab_ids = NULL;
    tok->vocab_size = 0;
    tok->merges = NULL;
    tok->num_merges = 0;
    tok->unk_token_id = 0;
    tok->pad_token_id = 1;
    tok->bos_token_id = 2;
    tok->eos_token_id = 3;
    
    return tok;
}

void tokenizer_free(Tokenizer* tok) {
    if (!tok) return;
    
    if (tok->vocab) {
        for (size_t i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    
    if (tok->vocab_ids) free(tok->vocab_ids);
    if (tok->merges) free(tok->merges);
    free(tok);
}

static int add_token(Tokenizer* tok, const char* token, size_t token_len) {
    if (tok->vocab_size >= MAX_VOCAB_SIZE) return -1;
    
    if (tok->vocab_size == 0) {
        tok->vocab = (char**)malloc(INITIAL_VOCAB_SIZE * sizeof(char*));
        tok->vocab_ids = (uint32_t*)malloc(INITIAL_VOCAB_SIZE * sizeof(uint32_t));
        if (!tok->vocab || !tok->vocab_ids) return -1;
    } else if (tok->vocab_size % INITIAL_VOCAB_SIZE == 0) {
        size_t new_size = tok->vocab_size + INITIAL_VOCAB_SIZE;
        tok->vocab = (char**)realloc(tok->vocab, new_size * sizeof(char*));
        tok->vocab_ids = (uint32_t*)realloc(tok->vocab_ids, new_size * sizeof(uint32_t));
        if (!tok->vocab || !tok->vocab_ids) return -1;
    }
    
    tok->vocab[tok->vocab_size] = (char*)malloc(token_len + 1);
    if (!tok->vocab[tok->vocab_size]) return -1;
    memcpy(tok->vocab[tok->vocab_size], token, token_len);
    tok->vocab[tok->vocab_size][token_len] = '\0';
    tok->vocab_ids[tok->vocab_size] = (uint32_t)tok->vocab_size;
    tok->vocab_size++;
    
    return 0;
}

static int init_basic_vocab(Tokenizer* tok) {
    add_token(tok, "<unk>", 5);
    add_token(tok, "<pad>", 5);
    add_token(tok, "<bos>", 5);
    add_token(tok, "<eos>", 5);
    
    for (int i = 0; i < 256; i++) {
        char c = (char)i;
        add_token(tok, &c, 1);
    }
    
    return 0;
}

static void count_words(const char* text, size_t text_len, 
                       char words[][MAX_WORD_LEN], size_t* word_count) {
    *word_count = 0;
    size_t word_idx = 0;
    size_t char_idx = 0;
    
    for (size_t i = 0; i < text_len && *word_count < 10000; i++) {
        if (isspace((unsigned char)text[i])) {
            if (char_idx > 0) {
                words[*word_count][char_idx] = '\0';
                (*word_count)++;
                char_idx = 0;
            }
        } else {
            if (char_idx < MAX_WORD_LEN - 1) {
                words[*word_count][char_idx++] = tolower((unsigned char)text[i]);
            }
        }
    }
    
    if (char_idx > 0) {
        words[*word_count][char_idx] = '\0';
        (*word_count)++;
    }
}

int tokenizer_train(Tokenizer* tok, const char* text, size_t text_len, size_t vocab_size) {
    if (!tok || !text) return -1;
    
    if (init_basic_vocab(tok) < 0) return -1;
    
    const char* common_bigrams[] = {
        "th", "he", "in", "er", "an", "re", "ed", "nd", "on", "en",
        "at", "ou", "it", "is", "or", "ti", "as", "to", "of", "te"
    };
    
    size_t num_bigrams = sizeof(common_bigrams) / sizeof(common_bigrams[0]);
    size_t target_merges = (vocab_size > tok->vocab_size) ? 
                          (vocab_size - tok->vocab_size) : 0;
    
    if (target_merges > num_bigrams) target_merges = num_bigrams;
    
    tok->merges = (uint32_t*)malloc(target_merges * 2 * sizeof(uint32_t));
    if (!tok->merges) return -1;
    
    for (size_t i = 0; i < target_merges && tok->vocab_size < vocab_size; i++) {
        const char* bigram = common_bigrams[i % num_bigrams];
        size_t bigram_len = strlen(bigram);
        
        if (add_token(tok, bigram, bigram_len) == 0) {
            tok->merges[tok->num_merges * 2] = (uint32_t)bigram[0];
            tok->merges[tok->num_merges * 2 + 1] = (uint32_t)bigram[1];
            tok->num_merges++;
        }
    }
    
    return 0;
}

static uint32_t find_token_id(Tokenizer* tok, const char* token, size_t token_len) {
    for (size_t i = 0; i < tok->vocab_size; i++) {
        if (strlen(tok->vocab[i]) == token_len && 
            memcmp(tok->vocab[i], token, token_len) == 0) {
            return tok->vocab_ids[i];
        }
    }
    return tok->unk_token_id;
}

int tokenizer_encode(Tokenizer* tok, const char* text, size_t text_len,
                     uint32_t* output, size_t* output_len, size_t max_output_len) {
    if (!tok || !text || !output || !output_len) return -1;
    
    *output_len = 0;
    
    size_t start = 0;
    for (size_t i = 0; i <= text_len && *output_len < max_output_len; i++) {
        if (i == text_len || isspace((unsigned char)text[i])) {
            if (i > start) {
                size_t word_len = i - start;
                uint32_t token_id = find_token_id(tok, text + start, word_len);
                
                if (token_id == tok->unk_token_id && word_len > 0) {
                    for (size_t j = start; j < i && *output_len < max_output_len; j++) {
                        char c = tolower((unsigned char)text[j]);
                        output[(*output_len)++] = find_token_id(tok, &c, 1);
                    }
                } else {
                    output[(*output_len)++] = token_id;
                }
            }
            start = i + 1;
        }
    }
    
    return 0;
}

int tokenizer_decode(Tokenizer* tok, const uint32_t* tokens, size_t num_tokens,
                     char* output, size_t* output_len, size_t max_output_len) {
    if (!tok || !tokens || !output || !output_len) return -1;
    
    *output_len = 0;
    
    for (size_t i = 0; i < num_tokens && *output_len < max_output_len - 1; i++) {
        uint32_t token_id = tokens[i];
        
        if (token_id == tok->unk_token_id || 
            token_id == tok->pad_token_id ||
            token_id == tok->bos_token_id ||
            token_id == tok->eos_token_id) {
            continue;
        }
        
        const char* token_str = NULL;
        for (size_t j = 0; j < tok->vocab_size; j++) {
            if (tok->vocab_ids[j] == token_id) {
                token_str = tok->vocab[j];
                break;
            }
        }
        
        if (token_str) {
            size_t token_str_len = strlen(token_str);
            if (*output_len + token_str_len < max_output_len) {
                memcpy(output + *output_len, token_str, token_str_len);
                *output_len += token_str_len;
            }
        }
    }
    
    output[*output_len] = '\0';
    return 0;
}

size_t tokenizer_vocab_size(Tokenizer* tok) {
    return tok ? tok->vocab_size : 0;
}

