#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    char** vocab;
    uint32_t* vocab_ids;
    size_t vocab_size;
    uint32_t* merges;
    size_t num_merges;
    uint32_t unk_token_id;
    uint32_t pad_token_id;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
} Tokenizer;

Tokenizer* tokenizer_new(void);

void tokenizer_free(Tokenizer* tok);

int tokenizer_train(Tokenizer* tok, const char* text, size_t text_len, size_t vocab_size);

int tokenizer_encode(Tokenizer* tok, const char* text, size_t text_len, 
                     uint32_t* output, size_t* output_len, size_t max_output_len);

int tokenizer_decode(Tokenizer* tok, const uint32_t* tokens, size_t num_tokens,
                     char* output, size_t* output_len, size_t max_output_len);

size_t tokenizer_vocab_size(Tokenizer* tok);

#endif

