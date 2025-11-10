import tokenizer

def test_tokenizer():
    print("Creating tokenizer...")
    tok = tokenizer.Tokenizer()
    
    training_text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Transformers revolutionized natural language processing.
    Python is a high-level programming language.
    Deep learning uses neural networks with multiple layers.
    """
    
    print(f"Training tokenizer on {len(training_text)} characters...")
    tok.train(training_text, vocab_size=500)
    
    vocab_size = tok.vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    test_texts = [
        "hello world",
        "the quick brown fox",
        "machine learning",
        "python programming"
    ]
    
    print("\n=== Encoding Tests ===")
    for text in test_texts:
        token_ids = tok.encode(text)
        print(f"Text: '{text}'")
        print(f"  Token IDs: {token_ids}")
        print(f"  Number of tokens: {len(token_ids)}")
    
    print("\n=== Decoding Tests ===")
    for text in test_texts:
        token_ids = tok.encode(text)
        decoded = tok.decode(token_ids)
        print(f"Original: '{text}'")
        print(f"Decoded:  '{decoded}'")
        print(f"Match: {text.lower() == decoded.lower()}")
        print()
    
    print("=== Long Text Test ===")
    long_text = "The transformer architecture has become the foundation of modern language models."
    token_ids = tok.encode(long_text)
    decoded = tok.decode(token_ids)
    print(f"Original length: {len(long_text)}")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"Original: '{long_text}'")
    print(f"Decoded:  '{decoded}'")
    
    print("\nTokenizer test completed!")

if __name__ == "__main__":
    try:
        test_tokenizer()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

