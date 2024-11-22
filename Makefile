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
MSCCL_MAX_NUM_STEPS ?= 64  # Default value for dynamic number of instructions
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
	${MAKE} -C src $* BUILDDIR=${ABSBUILDDIR} MSCCL_MAX_NUM_STEPS=${MSCCL_MAX_NUM_STEPS}

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR} MSCCL_MAX_NUM_STEPS=${MSCCL_MAX_NUM_STEPS}


pkg.debian.prep: lic
pkg.txz.prep: lic
