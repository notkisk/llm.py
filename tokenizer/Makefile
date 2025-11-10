CC=gcc
CFLAGS=-Wall -Wextra -O2 -fPIC -std=c11
PYTHON_CONFIG=python3-config
PYTHON_CFLAGS=$(shell $(PYTHON_CONFIG) --cflags)
PYTHON_LDFLAGS=$(shell $(PYTHON_CONFIG) --ldflags)

libtokenizer.so: tokenizer.c tokenizer.h
	$(CC) $(CFLAGS) -shared -o libtokenizer.so tokenizer.c

tokenizer.so: tokenizer.c tokenizer_python.c tokenizer.h
	$(CC) $(CFLAGS) $(PYTHON_CFLAGS) -shared -o tokenizer.so tokenizer.c tokenizer_python.c $(PYTHON_LDFLAGS)

install:
	python3 setup.py build_ext --inplace

clean:
	rm -f *.so *.o tokenizer.cpython-*.so
	rm -rf build/

test: install
	python3 test_tokenizer.py

.PHONY: clean test install

