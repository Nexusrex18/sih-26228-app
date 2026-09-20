CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror
LIBS = -lsodium -lsqlite3
all: libcvseal.a cvseal
cvseal.o: cvseal.c cvseal.h
	$(CC) $(CFLAGS) -c cvseal.c -o $@
libcvseal.a: cvseal.o
	ar rcs $@ $<
cvseal: cvseal_cli.c libcvseal.a
	$(CC) $(CFLAGS) cvseal_cli.c libcvseal.a $(LIBS) -o $@
clean:
	rm -f *.o *.a cvseal
