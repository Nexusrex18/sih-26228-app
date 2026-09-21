# syntax=docker/dockerfile:1
#
# The delivered image (Module E plan §7.9, D-E11, D-E12).
#
# Built ONCE on a networked machine (Mode A) from Backend's `cva-bundle.tar` — the boundary
# between the two seats is the tar. After that the image is `docker save`d, carried across
# the air gap, `docker load`ed, and never needs a network again.
#
# Build through `make -f docker/image.mk image`, never by hand: that is what passes the
# pinned base digest, the commit, and runs the size and no-CUDA assertions afterwards.

# The base is pinned by DIGEST, never by tag. `image.mk` reads it from docker/base.digest
# and refuses to build if that file is missing; the default below is deliberately not a
# valid reference, so a hand-run `docker build` fails loudly instead of floating on a tag.
ARG BASE_IMAGE=UNPINNED-run-make-f-docker-image.mk-pin-first

# ---------------------------------------------------------------------------- install
FROM ${BASE_IMAGE} AS install

ENV PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
# The bundle is copied into THIS stage only. Nothing from it but the installed venv and the
# source tree reaches the final image, so the ~400 MB tar is never a layer anyone ships.
COPY cva-bundle.tar /build/
# The source is moved to its FINAL path before the editable install, so the path the
# install records is the one the image runs from — nothing is rewritten afterwards.
RUN tar -xf cva-bundle.tar \
 && mv /build/cva-src /opt/cva-src \
 && mv /build/fixtures /opt/cva-fixtures \
 && mv /build/MANIFEST.json /opt/cva-src/MANIFEST.json \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-index --find-links=/build/wheelhouse \
        -r /build/requirements.lock \
 && /opt/venv/bin/pip install --no-index --no-build-isolation --no-deps -e /opt/cva-src \
 && rm -rf /build

# The dashboard's static export. It is gitignored, so it is not in the bundle's source
# snapshot; `image.mk` builds it and it is copied in beside the package that serves it
# (`cva/web/views/spa.py` resolves `<src>/frontend/out`).
COPY frontend/out /opt/cva-src/frontend/out

# ---------------------------------------------------------------------------- final
FROM ${BASE_IMAGE}

ARG CVA_CODE_COMMIT=unknown
LABEL org.opencontainers.image.title="cva" \
      org.opencontainers.image.description="CV Integrity Assurance — scanner, seal and analyst dashboard (SIH 26228)" \
      org.opencontainers.image.revision="${CVA_CODE_COMMIT}"

ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # No `.git` in an image; the CLI accepts this only if it is exactly 40 hex characters.
    CVA_CODE_COMMIT=${CVA_CODE_COMMIT} \
    # Where the reviewed coverage-standing file lives in the image.
    CVA_COVERAGE_STANDING=/opt/cva-src/docs/coverage-standing.yaml \
    # matplotlib and torch want a writable cache; the root fs is read-only, /tmp is tmpfs.
    MPLCONFIGDIR=/tmp/mpl \
    TORCH_HOME=/tmp/torch \
    HOME=/tmp

# Three service accounts sharing one group, so ledgerd's 0660 socket is reachable by the
# two callers and by nobody else. The uids are fixed because `docker/ledgerd-policy.json`
# names them: the daemon's allowlist is per uid (plan S10, §5.2).
#   10001 cva-ledgerd  holds the signing key; the only process that can sign
#   10002 cva-web      appends analyst_event, reads the ledger
#   10003 cva-scan     appends scan_record and nothing else
RUN groupadd --system --gid 10000 cva \
 && useradd --system --uid 10001 --gid cva --no-create-home --shell /usr/sbin/nologin cva-ledgerd \
 && useradd --system --uid 10002 --gid cva --no-create-home --shell /usr/sbin/nologin cva-web \
 && useradd --system --uid 10003 --gid cva --no-create-home --shell /usr/sbin/nologin cva-scan \
 && mkdir -p /var/lib/cva/reports /var/lib/cva/ledger /var/lib/cva/index /var/lib/cva/keys \
             /run/cva \
 && chown -R root:cva /var/lib/cva /run/cva \
 && chmod 0770 /var/lib/cva/reports /var/lib/cva/ledger /var/lib/cva/index /run/cva \
 # keys/ is a mount point only. The key is NEVER baked in (plan §7.9); it is mounted
 # read-only into the ledgerd container alone.
 && chmod 0750 /var/lib/cva/keys

COPY --from=install /opt/venv /opt/venv
COPY --from=install /opt/cva-src /opt/cva-src
COPY --from=install /opt/cva-fixtures /opt/cva-fixtures
COPY docker/entrypoint.sh /usr/local/bin/cva-entrypoint
COPY docker/ledgerd-policy.json /etc/cva/ledgerd-policy.json
RUN chmod 0555 /usr/local/bin/cva-entrypoint \
 && chmod 0444 /etc/cva/ledgerd-policy.json

WORKDIR /var/lib/cva
VOLUME ["/var/lib/cva/reports", "/var/lib/cva/ledger", "/var/lib/cva/index"]

# Non-root by default. The web and ledgerd roles are selected with `--user` as well as the
# entrypoint argument, so a container can never be both.
USER 10003:10000

EXPOSE 8713
ENTRYPOINT ["/usr/local/bin/cva-entrypoint"]
CMD ["selftest"]
