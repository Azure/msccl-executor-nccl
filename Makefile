#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.PHONY : all clean

default : src.build
install : src.install
BUILDDIR ?= $(abspath ./build)
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := src pkg
ENABLE_PRECISION_CLIPPING_HALF ?= 0 # Flag to enable precision flag for half, set 1 to enable, 0 to disable
clean: ${TARGETS:%=%.clean}
test.build: src.build
LICENSE_FILES := LICENSE.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(BUILDDIR)/%)
lic: $(LICENSE_TARGETS)

${BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${BUILDDIR}
	cp $< $@

src.%:
	${MAKE} -C src $* BUILDDIR=${ABSBUILDDIR} ENABLE_PRECISION_CLIPPING_HALF=${ENABLE_PRECISION_CLIPPING_HALF}

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR} ENABLE_PRECISION_CLIPPING_HALF=${ENABLE_PRECISION_CLIPPING_HALF}

pkg.debian.prep: lic
pkg.txz.prep: lic
