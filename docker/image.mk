# The delivered image (Module E plan §7.9). Run from the repo root:
#
#   make -f docker/image.mk pin        # Mode A, once: record the base image's digest
#   make -f docker/image.mk image      # bundle + dashboard + build + assertions
#   make -f docker/image.mk save       # cva-image.tar(.zst) for the air gap
#   make -f docker/image.mk check      # the assertions alone, against a built image
#   make -f docker/image.mk podman     # the same image through podman (rootless parity)
#
# On the air-gapped host only `docker load -i cva-image.tar` is needed. Nothing below runs
# there.

ENGINE     ?= docker
BASE_REPO  ?= python:3.12-slim-bookworm
DIGEST_FILE := docker/base.digest
IMAGE      ?= cva:latest
TAR        ?= cva-image.tar
# §7.9: the delivered image is < 4 GB (the bundle's own 3.5 GB is a different artefact).
LIMIT_BYTES := 4000000000
COMMIT     := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)

.PHONY: pin image build dashboard bundle check save podman

# The one step that needs a registry. It records the digest; every later build uses it, so
# the base can never float under a tag between two builds of "the same" image.
pin:
	$(ENGINE) pull $(BASE_REPO)
	$(ENGINE) image inspect --format '{{index .RepoDigests 0}}' $(BASE_REPO) \
		| sed 's/.*@//' > $(DIGEST_FILE)
	@echo "pinned $(BASE_REPO)@$$(cat $(DIGEST_FILE)) -> $(DIGEST_FILE); commit it"

dashboard:
	npm --prefix frontend ci
	npm --prefix frontend run build

bundle:
	$(MAKE) wheelhouse bundle

image: bundle dashboard build check

build:
	@test -s $(DIGEST_FILE) || { \
		echo "no $(DIGEST_FILE): run 'make -f docker/image.mk pin' once (Mode A)."; \
		echo "The base is pinned by digest, never by tag (§7.9)."; exit 1; }
	@test -f cva-bundle.tar || { echo "no cva-bundle.tar: run 'make bundle'"; exit 1; }
	@test -d frontend/out || { echo "no frontend/out: run the dashboard build"; exit 1; }
	$(ENGINE) build -f docker/Dockerfile \
		--build-arg BASE_IMAGE=$(BASE_REPO)@$$(cat $(DIGEST_FILE)) \
		--build-arg CVA_CODE_COMMIT=$(COMMIT) \
		-t $(IMAGE) .

# Assertions on the BUILT image, not on its inputs: a size or a package that only the
# final layers reveal is exactly what these exist to catch.
check:
	@size=$$($(ENGINE) image inspect --format '{{.Size}}' $(IMAGE)); \
	echo "$(IMAGE): $$size bytes"; \
	test "$$size" -lt $(LIMIT_BYTES) || { echo "image is not < 4 GB (§7.9)"; exit 1; }
	@! $(ENGINE) run --rm --network none --entrypoint /opt/venv/bin/pip $(IMAGE) \
		list --format=freeze | grep -Ei '^(nvidia|triton)' \
		|| { echo "a CUDA package entered the image (§5.7)"; exit 1; }
	@test "$$($(ENGINE) image inspect --format '{{.Config.User}}' $(IMAGE))" != "" \
		|| { echo "the image runs as root"; exit 1; }
	@! $(ENGINE) run --rm --network none --entrypoint /bin/sh $(IMAGE) \
		-c 'find / -xdev \( -name "*.key" -o -name "signing.key*" \) 2>/dev/null' | grep . \
		|| { echo "a key file is baked into a layer (§7.9)"; exit 1; }
	$(ENGINE) run --rm --network none --read-only --tmpfs /tmp --cap-drop ALL \
		--security-opt no-new-privileges $(IMAGE) selftest
	@echo "image checks passed"

save:
	$(ENGINE) save $(IMAGE) -o $(TAR)
	sha256sum $(TAR) > $(TAR).sha256
	@echo "wrote $(TAR) and $(TAR).sha256 — verify the hash on the far side of the gap first"

podman:
	$(MAKE) -f docker/image.mk ENGINE=podman build check
