savedcmd_mutant.mod.o := gcc -Wp,-MMD,./.mutant.mod.o.d -nostdinc -I/its/home/mm2350/Desktop/bbr/arch/x86/include -I/its/home/mm2350/Desktop/bbr/arch/x86/include/generated -I/its/home/mm2350/Desktop/bbr/include -I/its/home/mm2350/Desktop/bbr/include -I/its/home/mm2350/Desktop/bbr/arch/x86/include/uapi -I/its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi -I/its/home/mm2350/Desktop/bbr/include/uapi -I/its/home/mm2350/Desktop/bbr/include/generated/uapi -include /its/home/mm2350/Desktop/bbr/include/linux/compiler-version.h -include /its/home/mm2350/Desktop/bbr/include/linux/kconfig.h -include /its/home/mm2350/Desktop/bbr/include/linux/compiler_types.h -D__KERNEL__ -std=gnu11 -fshort-wchar -funsigned-char -fno-common -fno-PIE -fno-strict-aliasing -mno-sse -mno-mmx -mno-sse2 -mno-3dnow -mno-avx -fcf-protection=none -m64 -falign-jumps=1 -falign-loops=1 -mno-80387 -mno-fp-ret-in-387 -mpreferred-stack-boundary=3 -mskip-rax-setup -mtune=generic -mno-red-zone -mcmodel=kernel -Wno-sign-compare -fno-asynchronous-unwind-tables -mindirect-branch=thunk-extern -mindirect-branch-register -mindirect-branch-cs-prefix -mfunction-return=thunk-extern -fno-jump-tables -fpatchable-function-entry=16,16 -fno-delete-null-pointer-checks -O2 -fno-allow-store-data-races -fstack-protector-strong -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-stack-clash-protection -fzero-call-used-regs=used-gpr -pg -mrecord-mcount -mfentry -DCC_USING_FENTRY -falign-functions=16 -fno-strict-overflow -fno-stack-check -fconserve-stack -Wall -Wundef -Werror=implicit-function-declaration -Werror=implicit-int -Werror=return-type -Werror=strict-prototypes -Wno-format-security -Wno-trigraphs -Wno-frame-address -Wno-address-of-packed-member -Wmissing-declarations -Wmissing-prototypes -Wframe-larger-than=1024 -Wno-main -Wvla -Wno-pointer-sign -Wcast-function-type -Wno-stringop-overflow -Wno-array-bounds -Wno-alloc-size-larger-than -Wimplicit-fallthrough=5 -Werror=date-time -Werror=incompatible-pointer-types -Werror=designated-init -Wenum-conversion -Wextra -Wunused -Wno-unused-but-set-variable -Wno-unused-const-variable -Wno-packed-not-aligned -Wno-format-overflow -Wno-format-truncation -Wno-stringop-truncation -Wno-override-init -Wno-missing-field-initializers -Wno-type-limits -Wno-shift-negative-value -Wno-maybe-uninitialized -Wno-sign-compare -Wno-unused-parameter -g -gdwarf-5  -fsanitize=bounds-strict -fsanitize=shift -fsanitize=bool -fsanitize=enum  -fsanitize=signed-integer-overflow  -DMODULE  -DKBUILD_BASENAME='"mutant.mod"' -DKBUILD_MODNAME='"mutant"' -D__KBUILD_MODNAME=kmod_mutant -c -o mutant.mod.o mutant.mod.c   ; /its/home/mm2350/Desktop/bbr/tools/objtool/objtool --hacks=jump_label --hacks=noinstr --hacks=skylake --retpoline --rethunk --stackval --static-call --uaccess --prefix=16   --module mutant.mod.o

source_mutant.mod.o := mutant.mod.c

deps_mutant.mod.o := \
    $(wildcard include/config/MODULE_UNLOAD) \
  /its/home/mm2350/Desktop/bbr/include/linux/compiler-version.h \
    $(wildcard include/config/CC_VERSION_TEXT) \
  /its/home/mm2350/Desktop/bbr/include/linux/kconfig.h \
    $(wildcard include/config/CPU_BIG_ENDIAN) \
    $(wildcard include/config/BOOGER) \
    $(wildcard include/config/FOO) \
  /its/home/mm2350/Desktop/bbr/include/linux/compiler_types.h \
    $(wildcard include/config/DEBUG_INFO_BTF) \
    $(wildcard include/config/PAHOLE_HAS_BTF_TAG) \
    $(wildcard include/config/FUNCTION_ALIGNMENT) \
    $(wildcard include/config/CC_HAS_SANE_FUNCTION_ALIGNMENT) \
    $(wildcard include/config/X86_64) \
    $(wildcard include/config/ARM64) \
    $(wildcard include/config/LD_DEAD_CODE_DATA_ELIMINATION) \
    $(wildcard include/config/LTO_CLANG) \
    $(wildcard include/config/HAVE_ARCH_COMPILER_H) \
    $(wildcard include/config/CC_HAS_COUNTED_BY) \
    $(wildcard include/config/UBSAN_SIGNED_WRAP) \
    $(wildcard include/config/CC_HAS_ASM_INLINE) \
  /its/home/mm2350/Desktop/bbr/include/linux/compiler_attributes.h \
  /its/home/mm2350/Desktop/bbr/include/linux/compiler-gcc.h \
    $(wildcard include/config/MITIGATION_RETPOLINE) \
    $(wildcard include/config/ARCH_USE_BUILTIN_BSWAP) \
    $(wildcard include/config/SHADOW_CALL_STACK) \
    $(wildcard include/config/KCOV) \
  /its/home/mm2350/Desktop/bbr/include/linux/module.h \
    $(wildcard include/config/MODULES) \
    $(wildcard include/config/SYSFS) \
    $(wildcard include/config/MODULES_TREE_LOOKUP) \
    $(wildcard include/config/LIVEPATCH) \
    $(wildcard include/config/STACKTRACE_BUILD_ID) \
    $(wildcard include/config/ARCH_USES_CFI_TRAPS) \
    $(wildcard include/config/MODULE_SIG) \
    $(wildcard include/config/GENERIC_BUG) \
    $(wildcard include/config/KALLSYMS) \
    $(wildcard include/config/SMP) \
    $(wildcard include/config/TRACEPOINTS) \
    $(wildcard include/config/TREE_SRCU) \
    $(wildcard include/config/BPF_EVENTS) \
    $(wildcard include/config/DEBUG_INFO_BTF_MODULES) \
    $(wildcard include/config/JUMP_LABEL) \
    $(wildcard include/config/TRACING) \
    $(wildcard include/config/EVENT_TRACING) \
    $(wildcard include/config/FTRACE_MCOUNT_RECORD) \
    $(wildcard include/config/KPROBES) \
    $(wildcard include/config/HAVE_STATIC_CALL_INLINE) \
    $(wildcard include/config/KUNIT) \
    $(wildcard include/config/PRINTK_INDEX) \
    $(wildcard include/config/CONSTRUCTORS) \
    $(wildcard include/config/FUNCTION_ERROR_INJECTION) \
    $(wildcard include/config/DYNAMIC_DEBUG_CORE) \
    $(wildcard include/config/ARCH_HAS_EXECMEM_ROX) \
  /its/home/mm2350/Desktop/bbr/include/linux/list.h \
    $(wildcard include/config/LIST_HARDENED) \
    $(wildcard include/config/DEBUG_LIST) \
  /its/home/mm2350/Desktop/bbr/include/linux/container_of.h \
  /its/home/mm2350/Desktop/bbr/include/linux/build_bug.h \
  /its/home/mm2350/Desktop/bbr/include/linux/compiler.h \
    $(wildcard include/config/TRACE_BRANCH_PROFILING) \
    $(wildcard include/config/PROFILE_ALL_BRANCHES) \
    $(wildcard include/config/OBJTOOL) \
    $(wildcard include/config/64BIT) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/asm/rwonce.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/rwonce.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kasan-checks.h \
    $(wildcard include/config/KASAN_GENERIC) \
    $(wildcard include/config/KASAN_SW_TAGS) \
  /its/home/mm2350/Desktop/bbr/include/linux/types.h \
    $(wildcard include/config/HAVE_UID16) \
    $(wildcard include/config/UID16) \
    $(wildcard include/config/ARCH_DMA_ADDR_T_64BIT) \
    $(wildcard include/config/PHYS_ADDR_T_64BIT) \
    $(wildcard include/config/ARCH_32BIT_USTAT_F_TINODE) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/types.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/types.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/types.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/int-ll64.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/int-ll64.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/bitsperlong.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitsperlong.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/bitsperlong.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/posix_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/stddef.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/stddef.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/posix_types.h \
    $(wildcard include/config/X86_32) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/posix_types_64.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/posix_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kcsan-checks.h \
    $(wildcard include/config/KCSAN) \
    $(wildcard include/config/KCSAN_WEAK_MEMORY) \
    $(wildcard include/config/KCSAN_IGNORE_ATOMICS) \
  /its/home/mm2350/Desktop/bbr/include/linux/poison.h \
    $(wildcard include/config/ILLEGAL_POINTER_VALUE) \
  /its/home/mm2350/Desktop/bbr/include/linux/const.h \
  /its/home/mm2350/Desktop/bbr/include/vdso/const.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/const.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/barrier.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/alternative.h \
    $(wildcard include/config/CALL_THUNKS) \
  /its/home/mm2350/Desktop/bbr/include/linux/stringify.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/asm.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/extable_fixup_types.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/nops.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/barrier.h \
  /its/home/mm2350/Desktop/bbr/include/linux/stat.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/stat.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/stat.h \
  /its/home/mm2350/Desktop/bbr/include/linux/time.h \
    $(wildcard include/config/POSIX_TIMERS) \
  /its/home/mm2350/Desktop/bbr/include/linux/cache.h \
    $(wildcard include/config/ARCH_HAS_CACHE_LINE_SIZE) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/kernel.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/sysinfo.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/cache.h \
    $(wildcard include/config/X86_L1_CACHE_SHIFT) \
    $(wildcard include/config/X86_INTERNODE_CACHE_SHIFT) \
    $(wildcard include/config/X86_VSMP) \
  /its/home/mm2350/Desktop/bbr/include/linux/linkage.h \
    $(wildcard include/config/ARCH_USE_SYM_ANNOTATIONS) \
  /its/home/mm2350/Desktop/bbr/include/linux/export.h \
    $(wildcard include/config/MODVERSIONS) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/linkage.h \
    $(wildcard include/config/CALL_PADDING) \
    $(wildcard include/config/MITIGATION_RETHUNK) \
    $(wildcard include/config/MITIGATION_SLS) \
    $(wildcard include/config/FUNCTION_PADDING_BYTES) \
    $(wildcard include/config/UML) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/ibt.h \
    $(wildcard include/config/X86_KERNEL_IBT) \
  /its/home/mm2350/Desktop/bbr/include/linux/math64.h \
    $(wildcard include/config/ARCH_SUPPORTS_INT128) \
  /its/home/mm2350/Desktop/bbr/include/linux/math.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/div64.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/div64.h \
    $(wildcard include/config/CC_OPTIMIZE_FOR_PERFORMANCE) \
  /its/home/mm2350/Desktop/bbr/include/vdso/math64.h \
  /its/home/mm2350/Desktop/bbr/include/linux/time64.h \
  /its/home/mm2350/Desktop/bbr/include/vdso/time64.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/time.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/time_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/time32.h \
  /its/home/mm2350/Desktop/bbr/include/linux/timex.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/timex.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/param.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/param.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/param.h \
    $(wildcard include/config/HZ) \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/param.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/timex.h \
    $(wildcard include/config/X86_TSC) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/processor.h \
    $(wildcard include/config/X86_VMX_FEATURE_NAMES) \
    $(wildcard include/config/X86_IOPL_IOPERM) \
    $(wildcard include/config/STACKPROTECTOR) \
    $(wildcard include/config/VM86) \
    $(wildcard include/config/X86_USER_SHADOW_STACK) \
    $(wildcard include/config/USE_X86_SEG_SUPPORT) \
    $(wildcard include/config/PARAVIRT_XXL) \
    $(wildcard include/config/CPU_SUP_AMD) \
    $(wildcard include/config/XEN) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/processor-flags.h \
    $(wildcard include/config/MITIGATION_PAGE_TABLE_ISOLATION) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/processor-flags.h \
  /its/home/mm2350/Desktop/bbr/include/linux/mem_encrypt.h \
    $(wildcard include/config/ARCH_HAS_MEM_ENCRYPT) \
    $(wildcard include/config/AMD_MEM_ENCRYPT) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/mem_encrypt.h \
    $(wildcard include/config/X86_MEM_ENCRYPT) \
  /its/home/mm2350/Desktop/bbr/include/linux/init.h \
    $(wildcard include/config/MEMORY_HOTPLUG) \
    $(wildcard include/config/HAVE_ARCH_PREL32_RELOCATIONS) \
  /its/home/mm2350/Desktop/bbr/include/linux/cc_platform.h \
    $(wildcard include/config/ARCH_HAS_CC_PLATFORM) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/math_emu.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/ptrace.h \
    $(wildcard include/config/PARAVIRT) \
    $(wildcard include/config/IA32_EMULATION) \
    $(wildcard include/config/X86_DEBUGCTLMSR) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/segment.h \
    $(wildcard include/config/XEN_PV) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/page_types.h \
    $(wildcard include/config/PHYSICAL_START) \
    $(wildcard include/config/PHYSICAL_ALIGN) \
    $(wildcard include/config/DYNAMIC_PHYSICAL_MASK) \
  /its/home/mm2350/Desktop/bbr/include/vdso/page.h \
    $(wildcard include/config/PAGE_SHIFT) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/page_64_types.h \
    $(wildcard include/config/KASAN) \
    $(wildcard include/config/DYNAMIC_MEMORY_LAYOUT) \
    $(wildcard include/config/X86_5LEVEL) \
    $(wildcard include/config/RANDOMIZE_BASE) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/kaslr.h \
    $(wildcard include/config/RANDOMIZE_MEMORY) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/ptrace.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/ptrace-abi.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/paravirt_types.h \
    $(wildcard include/config/PGTABLE_LEVELS) \
    $(wildcard include/config/ZERO_CALL_USED_REGS) \
    $(wildcard include/config/PARAVIRT_DEBUG) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/desc_defs.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/pgtable_types.h \
    $(wildcard include/config/X86_INTEL_MEMORY_PROTECTION_KEYS) \
    $(wildcard include/config/X86_PAE) \
    $(wildcard include/config/MEM_SOFT_DIRTY) \
    $(wildcard include/config/HAVE_ARCH_USERFAULTFD_WP) \
    $(wildcard include/config/PROC_FS) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/pgtable_64_types.h \
    $(wildcard include/config/KMSAN) \
    $(wildcard include/config/DEBUG_KMAP_LOCAL_FORCE_MAP) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/sparsemem.h \
    $(wildcard include/config/SPARSEMEM) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/nospec-branch.h \
    $(wildcard include/config/CALL_THUNKS_DEBUG) \
    $(wildcard include/config/MITIGATION_CALL_DEPTH_TRACKING) \
    $(wildcard include/config/NOINSTR_VALIDATION) \
    $(wildcard include/config/MITIGATION_UNRET_ENTRY) \
    $(wildcard include/config/MITIGATION_SRSO) \
    $(wildcard include/config/MITIGATION_IBPB_ENTRY) \
  /its/home/mm2350/Desktop/bbr/include/linux/static_key.h \
  /its/home/mm2350/Desktop/bbr/include/linux/jump_label.h \
    $(wildcard include/config/HAVE_ARCH_JUMP_LABEL_RELATIVE) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/jump_label.h \
    $(wildcard include/config/HAVE_JUMP_LABEL_HACK) \
  /its/home/mm2350/Desktop/bbr/include/linux/objtool.h \
    $(wildcard include/config/FRAME_POINTER) \
  /its/home/mm2350/Desktop/bbr/include/linux/objtool_types.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/cpufeatures.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/required-features.h \
    $(wildcard include/config/X86_MINIMUM_CPU_FAMILY) \
    $(wildcard include/config/MATH_EMULATION) \
    $(wildcard include/config/X86_CMPXCHG64) \
    $(wildcard include/config/X86_CMOV) \
    $(wildcard include/config/X86_P6_NOP) \
    $(wildcard include/config/MATOM) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/disabled-features.h \
    $(wildcard include/config/X86_UMIP) \
    $(wildcard include/config/ADDRESS_MASKING) \
    $(wildcard include/config/INTEL_IOMMU_SVM) \
    $(wildcard include/config/X86_SGX) \
    $(wildcard include/config/INTEL_TDX_GUEST) \
    $(wildcard include/config/X86_FRED) \
    $(wildcard include/config/KVM_AMD_SEV) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/msr-index.h \
  /its/home/mm2350/Desktop/bbr/include/linux/bits.h \
  /its/home/mm2350/Desktop/bbr/include/vdso/bits.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/bits.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/unwind_hints.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/orc_types.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/byteorder.h \
  /its/home/mm2350/Desktop/bbr/include/linux/byteorder/little_endian.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/byteorder/little_endian.h \
  /its/home/mm2350/Desktop/bbr/include/linux/swab.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/swab.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/swab.h \
  /its/home/mm2350/Desktop/bbr/include/linux/byteorder/generic.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/percpu.h \
    $(wildcard include/config/X86_64_SMP) \
    $(wildcard include/config/CC_HAS_NAMED_AS) \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/percpu.h \
    $(wildcard include/config/DEBUG_PREEMPT) \
    $(wildcard include/config/HAVE_SETUP_PER_CPU_AREA) \
  /its/home/mm2350/Desktop/bbr/include/linux/threads.h \
    $(wildcard include/config/NR_CPUS) \
    $(wildcard include/config/BASE_SMALL) \
  /its/home/mm2350/Desktop/bbr/include/linux/percpu-defs.h \
    $(wildcard include/config/DEBUG_FORCE_WEAK_PER_CPU) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/current.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/asm-offsets.h \
  /its/home/mm2350/Desktop/bbr/include/generated/asm-offsets.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/GEN-for-each-reg.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/spinlock_types.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/qspinlock_types.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/qrwlock_types.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/proto.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/ldt.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/sigcontext.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/cpuid.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/string.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/string_64.h \
    $(wildcard include/config/ARCH_HAS_UACCESS_FLUSHCACHE) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/paravirt.h \
    $(wildcard include/config/PARAVIRT_SPINLOCKS) \
    $(wildcard include/config/DEBUG_ENTRY) \
  /its/home/mm2350/Desktop/bbr/include/linux/bug.h \
    $(wildcard include/config/BUG_ON_DATA_CORRUPTION) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/bug.h \
    $(wildcard include/config/DEBUG_BUGVERBOSE) \
  /its/home/mm2350/Desktop/bbr/include/linux/instrumentation.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bug.h \
    $(wildcard include/config/BUG) \
    $(wildcard include/config/GENERIC_BUG_RELATIVE_POINTERS) \
  /its/home/mm2350/Desktop/bbr/include/linux/once_lite.h \
  /its/home/mm2350/Desktop/bbr/include/linux/panic.h \
    $(wildcard include/config/PANIC_TIMEOUT) \
  /its/home/mm2350/Desktop/bbr/include/linux/printk.h \
    $(wildcard include/config/MESSAGE_LOGLEVEL_DEFAULT) \
    $(wildcard include/config/CONSOLE_LOGLEVEL_DEFAULT) \
    $(wildcard include/config/CONSOLE_LOGLEVEL_QUIET) \
    $(wildcard include/config/EARLY_PRINTK) \
    $(wildcard include/config/PRINTK) \
    $(wildcard include/config/DYNAMIC_DEBUG) \
  /its/home/mm2350/Desktop/bbr/include/linux/stdarg.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kern_levels.h \
  /its/home/mm2350/Desktop/bbr/include/linux/ratelimit_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/spinlock_types_raw.h \
    $(wildcard include/config/DEBUG_SPINLOCK) \
    $(wildcard include/config/DEBUG_LOCK_ALLOC) \
  /its/home/mm2350/Desktop/bbr/include/linux/lockdep_types.h \
    $(wildcard include/config/PROVE_RAW_LOCK_NESTING) \
    $(wildcard include/config/LOCKDEP) \
    $(wildcard include/config/LOCK_STAT) \
  /its/home/mm2350/Desktop/bbr/include/linux/dynamic_debug.h \
  /its/home/mm2350/Desktop/bbr/include/linux/cpumask.h \
    $(wildcard include/config/FORCE_NR_CPUS) \
    $(wildcard include/config/HOTPLUG_CPU) \
    $(wildcard include/config/DEBUG_PER_CPU_MAPS) \
    $(wildcard include/config/CPUMASK_OFFSTACK) \
  /its/home/mm2350/Desktop/bbr/include/linux/cleanup.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kernel.h \
    $(wildcard include/config/PREEMPT_VOLUNTARY_BUILD) \
    $(wildcard include/config/PREEMPT_DYNAMIC) \
    $(wildcard include/config/HAVE_PREEMPT_DYNAMIC_CALL) \
    $(wildcard include/config/HAVE_PREEMPT_DYNAMIC_KEY) \
    $(wildcard include/config/PREEMPT_) \
    $(wildcard include/config/DEBUG_ATOMIC_SLEEP) \
    $(wildcard include/config/MMU) \
    $(wildcard include/config/PROVE_LOCKING) \
  /its/home/mm2350/Desktop/bbr/include/linux/align.h \
  /its/home/mm2350/Desktop/bbr/include/linux/array_size.h \
  /its/home/mm2350/Desktop/bbr/include/linux/limits.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/limits.h \
  /its/home/mm2350/Desktop/bbr/include/vdso/limits.h \
  /its/home/mm2350/Desktop/bbr/include/linux/bitops.h \
  /its/home/mm2350/Desktop/bbr/include/linux/typecheck.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitops/generic-non-atomic.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/bitops.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/rmwcc.h \
  /its/home/mm2350/Desktop/bbr/include/linux/args.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitops/sched.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/arch_hweight.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitops/const_hweight.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitops/instrumented-atomic.h \
  /its/home/mm2350/Desktop/bbr/include/linux/instrumented.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kmsan-checks.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitops/instrumented-non-atomic.h \
    $(wildcard include/config/KCSAN_ASSUME_PLAIN_WRITES_ATOMIC) \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitops/instrumented-lock.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitops/le.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/bitops/ext2-atomic-setbit.h \
  /its/home/mm2350/Desktop/bbr/include/linux/hex.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kstrtox.h \
  /its/home/mm2350/Desktop/bbr/include/linux/log2.h \
    $(wildcard include/config/ARCH_HAS_ILOG2_U32) \
    $(wildcard include/config/ARCH_HAS_ILOG2_U64) \
  /its/home/mm2350/Desktop/bbr/include/linux/minmax.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sprintf.h \
  /its/home/mm2350/Desktop/bbr/include/linux/static_call_types.h \
    $(wildcard include/config/HAVE_STATIC_CALL) \
  /its/home/mm2350/Desktop/bbr/include/linux/instruction_pointer.h \
  /its/home/mm2350/Desktop/bbr/include/linux/wordpart.h \
  /its/home/mm2350/Desktop/bbr/include/linux/bitmap.h \
  /its/home/mm2350/Desktop/bbr/include/linux/errno.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/errno.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/errno.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/errno.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/errno-base.h \
  /its/home/mm2350/Desktop/bbr/include/linux/find.h \
  /its/home/mm2350/Desktop/bbr/include/linux/string.h \
    $(wildcard include/config/BINARY_PRINTF) \
    $(wildcard include/config/FORTIFY_SOURCE) \
  /its/home/mm2350/Desktop/bbr/include/linux/err.h \
  /its/home/mm2350/Desktop/bbr/include/linux/overflow.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/string.h \
  /its/home/mm2350/Desktop/bbr/include/linux/fortify-string.h \
    $(wildcard include/config/CC_HAS_KASAN_MEMINTRINSIC_PREFIX) \
    $(wildcard include/config/GENERIC_ENTRY) \
  /its/home/mm2350/Desktop/bbr/include/linux/bitfield.h \
  /its/home/mm2350/Desktop/bbr/include/linux/bitmap-str.h \
  /its/home/mm2350/Desktop/bbr/include/linux/cpumask_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/atomic.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/atomic.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/cmpxchg.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/cmpxchg_64.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/atomic64_64.h \
  /its/home/mm2350/Desktop/bbr/include/linux/atomic/atomic-arch-fallback.h \
    $(wildcard include/config/GENERIC_ATOMIC64) \
  /its/home/mm2350/Desktop/bbr/include/linux/atomic/atomic-long.h \
  /its/home/mm2350/Desktop/bbr/include/linux/atomic/atomic-instrumented.h \
  /its/home/mm2350/Desktop/bbr/include/linux/gfp_types.h \
    $(wildcard include/config/KASAN_HW_TAGS) \
    $(wildcard include/config/SLAB_OBJ_EXT) \
  /its/home/mm2350/Desktop/bbr/include/linux/numa.h \
    $(wildcard include/config/NODES_SHIFT) \
    $(wildcard include/config/NUMA_KEEP_MEMINFO) \
    $(wildcard include/config/NUMA) \
    $(wildcard include/config/HAVE_ARCH_NODE_DEV_GROUP) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/frame.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/page.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/page_64.h \
    $(wildcard include/config/DEBUG_VIRTUAL) \
    $(wildcard include/config/X86_VSYSCALL_EMULATION) \
  /its/home/mm2350/Desktop/bbr/include/linux/range.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/memory_model.h \
    $(wildcard include/config/FLATMEM) \
    $(wildcard include/config/SPARSEMEM_VMEMMAP) \
  /its/home/mm2350/Desktop/bbr/include/linux/pfn.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/getorder.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/special_insns.h \
  /its/home/mm2350/Desktop/bbr/include/linux/irqflags.h \
    $(wildcard include/config/TRACE_IRQFLAGS) \
    $(wildcard include/config/PREEMPT_RT) \
    $(wildcard include/config/IRQSOFF_TRACER) \
    $(wildcard include/config/PREEMPT_TRACER) \
    $(wildcard include/config/DEBUG_IRQFLAGS) \
    $(wildcard include/config/TRACE_IRQFLAGS_SUPPORT) \
  /its/home/mm2350/Desktop/bbr/include/linux/irqflags_types.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/irqflags.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/fpu/types.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/vmxfeatures.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/vdso/processor.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/shstk.h \
  /its/home/mm2350/Desktop/bbr/include/linux/personality.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/personality.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/tsc.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/cpufeature.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/msr.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/cpumask.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/msr.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/ioctl.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/ioctl.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/ioctl.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/ioctl.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/shared/msr.h \
  /its/home/mm2350/Desktop/bbr/include/linux/percpu.h \
    $(wildcard include/config/MEM_ALLOC_PROFILING) \
    $(wildcard include/config/RANDOM_KMALLOC_CACHES) \
    $(wildcard include/config/PAGE_SIZE_4KB) \
    $(wildcard include/config/NEED_PER_CPU_PAGE_FIRST_CHUNK) \
  /its/home/mm2350/Desktop/bbr/include/linux/alloc_tag.h \
    $(wildcard include/config/MEM_ALLOC_PROFILING_DEBUG) \
    $(wildcard include/config/MEM_ALLOC_PROFILING_ENABLED_BY_DEFAULT) \
  /its/home/mm2350/Desktop/bbr/include/linux/codetag.h \
    $(wildcard include/config/CODE_TAGGING) \
  /its/home/mm2350/Desktop/bbr/include/linux/preempt.h \
    $(wildcard include/config/PREEMPT_COUNT) \
    $(wildcard include/config/TRACE_PREEMPT_TOGGLE) \
    $(wildcard include/config/PREEMPTION) \
    $(wildcard include/config/PREEMPT_NOTIFIERS) \
    $(wildcard include/config/PREEMPT_NONE) \
    $(wildcard include/config/PREEMPT_VOLUNTARY) \
    $(wildcard include/config/PREEMPT) \
    $(wildcard include/config/PREEMPT_LAZY) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/preempt.h \
  /its/home/mm2350/Desktop/bbr/include/linux/smp.h \
    $(wildcard include/config/UP_LATE_INIT) \
    $(wildcard include/config/CSD_LOCK_WAIT_DEBUG) \
  /its/home/mm2350/Desktop/bbr/include/linux/smp_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/llist.h \
    $(wildcard include/config/ARCH_HAVE_NMI_SAFE_CMPXCHG) \
  /its/home/mm2350/Desktop/bbr/include/linux/thread_info.h \
    $(wildcard include/config/THREAD_INFO_IN_TASK) \
    $(wildcard include/config/ARCH_HAS_PREEMPT_LAZY) \
    $(wildcard include/config/HAVE_ARCH_WITHIN_STACK_FRAMES) \
    $(wildcard include/config/HARDENED_USERCOPY) \
    $(wildcard include/config/SH) \
  /its/home/mm2350/Desktop/bbr/include/linux/restart_block.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/thread_info.h \
    $(wildcard include/config/COMPAT) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/smp.h \
    $(wildcard include/config/DEBUG_NMI_SELFTEST) \
  /its/home/mm2350/Desktop/bbr/include/linux/mmdebug.h \
    $(wildcard include/config/DEBUG_VM) \
    $(wildcard include/config/DEBUG_VM_IRQSOFF) \
    $(wildcard include/config/DEBUG_VM_PGFLAGS) \
  /its/home/mm2350/Desktop/bbr/include/linux/sched.h \
    $(wildcard include/config/VIRT_CPU_ACCOUNTING_NATIVE) \
    $(wildcard include/config/SCHED_INFO) \
    $(wildcard include/config/SCHEDSTATS) \
    $(wildcard include/config/SCHED_CORE) \
    $(wildcard include/config/FAIR_GROUP_SCHED) \
    $(wildcard include/config/RT_GROUP_SCHED) \
    $(wildcard include/config/RT_MUTEXES) \
    $(wildcard include/config/UCLAMP_TASK) \
    $(wildcard include/config/UCLAMP_BUCKETS_COUNT) \
    $(wildcard include/config/KMAP_LOCAL) \
    $(wildcard include/config/SCHED_CLASS_EXT) \
    $(wildcard include/config/CGROUP_SCHED) \
    $(wildcard include/config/BLK_DEV_IO_TRACE) \
    $(wildcard include/config/PREEMPT_RCU) \
    $(wildcard include/config/TASKS_RCU) \
    $(wildcard include/config/TASKS_TRACE_RCU) \
    $(wildcard include/config/MEMCG_V1) \
    $(wildcard include/config/LRU_GEN) \
    $(wildcard include/config/COMPAT_BRK) \
    $(wildcard include/config/CGROUPS) \
    $(wildcard include/config/BLK_CGROUP) \
    $(wildcard include/config/PSI) \
    $(wildcard include/config/PAGE_OWNER) \
    $(wildcard include/config/EVENTFD) \
    $(wildcard include/config/ARCH_HAS_CPU_PASID) \
    $(wildcard include/config/X86_BUS_LOCK_DETECT) \
    $(wildcard include/config/TASK_DELAY_ACCT) \
    $(wildcard include/config/ARCH_HAS_SCALED_CPUTIME) \
    $(wildcard include/config/VIRT_CPU_ACCOUNTING_GEN) \
    $(wildcard include/config/NO_HZ_FULL) \
    $(wildcard include/config/POSIX_CPUTIMERS) \
    $(wildcard include/config/POSIX_CPU_TIMERS_TASK_WORK) \
    $(wildcard include/config/KEYS) \
    $(wildcard include/config/SYSVIPC) \
    $(wildcard include/config/DETECT_HUNG_TASK) \
    $(wildcard include/config/IO_URING) \
    $(wildcard include/config/AUDIT) \
    $(wildcard include/config/AUDITSYSCALL) \
    $(wildcard include/config/DEBUG_MUTEXES) \
    $(wildcard include/config/UBSAN) \
    $(wildcard include/config/UBSAN_TRAP) \
    $(wildcard include/config/COMPACTION) \
    $(wildcard include/config/TASK_XACCT) \
    $(wildcard include/config/CPUSETS) \
    $(wildcard include/config/X86_CPU_RESCTRL) \
    $(wildcard include/config/FUTEX) \
    $(wildcard include/config/PERF_EVENTS) \
    $(wildcard include/config/NUMA_BALANCING) \
    $(wildcard include/config/RSEQ) \
    $(wildcard include/config/SCHED_MM_CID) \
    $(wildcard include/config/FAULT_INJECTION) \
    $(wildcard include/config/LATENCYTOP) \
    $(wildcard include/config/FUNCTION_GRAPH_TRACER) \
    $(wildcard include/config/MEMCG) \
    $(wildcard include/config/UPROBES) \
    $(wildcard include/config/BCACHE) \
    $(wildcard include/config/VMAP_STACK) \
    $(wildcard include/config/SECURITY) \
    $(wildcard include/config/BPF_SYSCALL) \
    $(wildcard include/config/GCC_PLUGIN_STACKLEAK) \
    $(wildcard include/config/X86_MCE) \
    $(wildcard include/config/KRETPROBES) \
    $(wildcard include/config/RETHOOK) \
    $(wildcard include/config/ARCH_HAS_PARANOID_L1D_FLUSH) \
    $(wildcard include/config/RV) \
    $(wildcard include/config/USER_EVENTS) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/sched.h \
  /its/home/mm2350/Desktop/bbr/include/linux/pid_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sem_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/shm.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/shmparam.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kmsan_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/mutex_types.h \
    $(wildcard include/config/MUTEX_SPIN_ON_OWNER) \
  /its/home/mm2350/Desktop/bbr/include/linux/osq_lock.h \
  /its/home/mm2350/Desktop/bbr/include/linux/spinlock_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rwlock_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/plist_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/hrtimer_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/timerqueue_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rbtree_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/timer_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/seccomp_types.h \
    $(wildcard include/config/SECCOMP) \
  /its/home/mm2350/Desktop/bbr/include/linux/nodemask_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/refcount_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/resource.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/resource.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/resource.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/resource.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/resource.h \
  /its/home/mm2350/Desktop/bbr/include/linux/latencytop.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/prio.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/signal_types.h \
    $(wildcard include/config/OLD_SIGACTION) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/signal.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/signal.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/signal.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/signal-defs.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/siginfo.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/siginfo.h \
  /its/home/mm2350/Desktop/bbr/include/linux/syscall_user_dispatch_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/mm_types_task.h \
    $(wildcard include/config/ARCH_WANT_BATCHED_UNMAP_TLB_FLUSH) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/tlbbatch.h \
  /its/home/mm2350/Desktop/bbr/include/linux/netdevice_xmit.h \
    $(wildcard include/config/NET_EGRESS) \
  /its/home/mm2350/Desktop/bbr/include/linux/task_io_accounting.h \
    $(wildcard include/config/TASK_IO_ACCOUNTING) \
  /its/home/mm2350/Desktop/bbr/include/linux/posix-timers_types.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/rseq.h \
  /its/home/mm2350/Desktop/bbr/include/linux/seqlock_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kcsan.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rv.h \
    $(wildcard include/config/RV_REACTORS) \
  /its/home/mm2350/Desktop/bbr/include/linux/livepatch_sched.h \
  /its/home/mm2350/Desktop/bbr/include/linux/uidgid_types.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/asm/kmap_size.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/kmap_size.h \
    $(wildcard include/config/DEBUG_KMAP_LOCAL) \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/ext.h \
    $(wildcard include/config/EXT_GROUP_SCHED) \
  /its/home/mm2350/Desktop/bbr/include/linux/spinlock.h \
  /its/home/mm2350/Desktop/bbr/include/linux/bottom_half.h \
  /its/home/mm2350/Desktop/bbr/include/linux/lockdep.h \
    $(wildcard include/config/DEBUG_LOCKING_API_SELFTESTS) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/asm/mmiowb.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/mmiowb.h \
    $(wildcard include/config/MMIOWB) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/spinlock.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/qspinlock.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/qspinlock.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/qrwlock.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/qrwlock.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rwlock.h \
  /its/home/mm2350/Desktop/bbr/include/linux/spinlock_api_smp.h \
    $(wildcard include/config/INLINE_SPIN_LOCK) \
    $(wildcard include/config/INLINE_SPIN_LOCK_BH) \
    $(wildcard include/config/INLINE_SPIN_LOCK_IRQ) \
    $(wildcard include/config/INLINE_SPIN_LOCK_IRQSAVE) \
    $(wildcard include/config/INLINE_SPIN_TRYLOCK) \
    $(wildcard include/config/INLINE_SPIN_TRYLOCK_BH) \
    $(wildcard include/config/UNINLINE_SPIN_UNLOCK) \
    $(wildcard include/config/INLINE_SPIN_UNLOCK_BH) \
    $(wildcard include/config/INLINE_SPIN_UNLOCK_IRQ) \
    $(wildcard include/config/INLINE_SPIN_UNLOCK_IRQRESTORE) \
    $(wildcard include/config/GENERIC_LOCKBREAK) \
  /its/home/mm2350/Desktop/bbr/include/linux/rwlock_api_smp.h \
    $(wildcard include/config/INLINE_READ_LOCK) \
    $(wildcard include/config/INLINE_WRITE_LOCK) \
    $(wildcard include/config/INLINE_READ_LOCK_BH) \
    $(wildcard include/config/INLINE_WRITE_LOCK_BH) \
    $(wildcard include/config/INLINE_READ_LOCK_IRQ) \
    $(wildcard include/config/INLINE_WRITE_LOCK_IRQ) \
    $(wildcard include/config/INLINE_READ_LOCK_IRQSAVE) \
    $(wildcard include/config/INLINE_WRITE_LOCK_IRQSAVE) \
    $(wildcard include/config/INLINE_READ_TRYLOCK) \
    $(wildcard include/config/INLINE_WRITE_TRYLOCK) \
    $(wildcard include/config/INLINE_READ_UNLOCK) \
    $(wildcard include/config/INLINE_WRITE_UNLOCK) \
    $(wildcard include/config/INLINE_READ_UNLOCK_BH) \
    $(wildcard include/config/INLINE_WRITE_UNLOCK_BH) \
    $(wildcard include/config/INLINE_READ_UNLOCK_IRQ) \
    $(wildcard include/config/INLINE_WRITE_UNLOCK_IRQ) \
    $(wildcard include/config/INLINE_READ_UNLOCK_IRQRESTORE) \
    $(wildcard include/config/INLINE_WRITE_UNLOCK_IRQRESTORE) \
  /its/home/mm2350/Desktop/bbr/include/linux/tracepoint-defs.h \
  /its/home/mm2350/Desktop/bbr/include/vdso/time32.h \
  /its/home/mm2350/Desktop/bbr/include/vdso/time.h \
  /its/home/mm2350/Desktop/bbr/include/linux/uidgid.h \
    $(wildcard include/config/MULTIUSER) \
    $(wildcard include/config/USER_NS) \
  /its/home/mm2350/Desktop/bbr/include/linux/highuid.h \
  /its/home/mm2350/Desktop/bbr/include/linux/buildid.h \
    $(wildcard include/config/VMCORE_INFO) \
  /its/home/mm2350/Desktop/bbr/include/linux/kmod.h \
  /its/home/mm2350/Desktop/bbr/include/linux/umh.h \
  /its/home/mm2350/Desktop/bbr/include/linux/gfp.h \
    $(wildcard include/config/HIGHMEM) \
    $(wildcard include/config/ZONE_DMA) \
    $(wildcard include/config/ZONE_DMA32) \
    $(wildcard include/config/ZONE_DEVICE) \
    $(wildcard include/config/CONTIG_ALLOC) \
  /its/home/mm2350/Desktop/bbr/include/linux/mmzone.h \
    $(wildcard include/config/ARCH_FORCE_MAX_ORDER) \
    $(wildcard include/config/CMA) \
    $(wildcard include/config/MEMORY_ISOLATION) \
    $(wildcard include/config/ZSMALLOC) \
    $(wildcard include/config/UNACCEPTED_MEMORY) \
    $(wildcard include/config/IOMMU_SUPPORT) \
    $(wildcard include/config/SWAP) \
    $(wildcard include/config/HUGETLB_PAGE) \
    $(wildcard include/config/TRANSPARENT_HUGEPAGE) \
    $(wildcard include/config/LRU_GEN_STATS) \
    $(wildcard include/config/LRU_GEN_WALKS_MMU) \
    $(wildcard include/config/MEMORY_FAILURE) \
    $(wildcard include/config/PAGE_EXTENSION) \
    $(wildcard include/config/DEFERRED_STRUCT_PAGE_INIT) \
    $(wildcard include/config/HAVE_MEMORYLESS_NODES) \
    $(wildcard include/config/SPARSEMEM_EXTREME) \
    $(wildcard include/config/HAVE_ARCH_PFN_VALID) \
  /its/home/mm2350/Desktop/bbr/include/linux/list_nulls.h \
  /its/home/mm2350/Desktop/bbr/include/linux/wait.h \
  /its/home/mm2350/Desktop/bbr/include/linux/seqlock.h \
  /its/home/mm2350/Desktop/bbr/include/linux/mutex.h \
  /its/home/mm2350/Desktop/bbr/include/linux/debug_locks.h \
  /its/home/mm2350/Desktop/bbr/include/linux/nodemask.h \
  /its/home/mm2350/Desktop/bbr/include/linux/random.h \
    $(wildcard include/config/VMGENID) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/random.h \
  /its/home/mm2350/Desktop/bbr/include/linux/irqnr.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/irqnr.h \
  /its/home/mm2350/Desktop/bbr/include/linux/pageblock-flags.h \
    $(wildcard include/config/HUGETLB_PAGE_SIZE_VARIABLE) \
  /its/home/mm2350/Desktop/bbr/include/linux/page-flags-layout.h \
  /its/home/mm2350/Desktop/bbr/include/generated/bounds.h \
  /its/home/mm2350/Desktop/bbr/include/linux/mm_types.h \
    $(wildcard include/config/HAVE_ALIGNED_STRUCT_PAGE) \
    $(wildcard include/config/HUGETLB_PMD_PAGE_TABLE_SHARING) \
    $(wildcard include/config/USERFAULTFD) \
    $(wildcard include/config/ANON_VMA_NAME) \
    $(wildcard include/config/PER_VMA_LOCK) \
    $(wildcard include/config/HAVE_ARCH_COMPAT_MMAP_BASES) \
    $(wildcard include/config/MEMBARRIER) \
    $(wildcard include/config/AIO) \
    $(wildcard include/config/MMU_NOTIFIER) \
    $(wildcard include/config/SPLIT_PMD_PTLOCKS) \
    $(wildcard include/config/IOMMU_MM_DATA) \
    $(wildcard include/config/KSM) \
    $(wildcard include/config/CORE_DUMP_DEFAULT_ELF_HEADERS) \
  /its/home/mm2350/Desktop/bbr/include/linux/auxvec.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/auxvec.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/auxvec.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kref.h \
  /its/home/mm2350/Desktop/bbr/include/linux/refcount.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rbtree.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rcupdate.h \
    $(wildcard include/config/TINY_RCU) \
    $(wildcard include/config/RCU_STRICT_GRACE_PERIOD) \
    $(wildcard include/config/RCU_LAZY) \
    $(wildcard include/config/TASKS_RCU_GENERIC) \
    $(wildcard include/config/RCU_STALL_COMMON) \
    $(wildcard include/config/KVM_XFER_TO_GUEST_WORK) \
    $(wildcard include/config/RCU_NOCB_CPU) \
    $(wildcard include/config/TASKS_RUDE_RCU) \
    $(wildcard include/config/TREE_RCU) \
    $(wildcard include/config/DEBUG_OBJECTS_RCU_HEAD) \
    $(wildcard include/config/PROVE_RCU) \
    $(wildcard include/config/ARCH_WEAK_RELEASE_ACQUIRE) \
  /its/home/mm2350/Desktop/bbr/include/linux/context_tracking_irq.h \
    $(wildcard include/config/CONTEXT_TRACKING_IDLE) \
  /its/home/mm2350/Desktop/bbr/include/linux/rcutree.h \
  /its/home/mm2350/Desktop/bbr/include/linux/maple_tree.h \
    $(wildcard include/config/MAPLE_RCU_DISABLED) \
    $(wildcard include/config/DEBUG_MAPLE_TREE) \
  /its/home/mm2350/Desktop/bbr/include/linux/rwsem.h \
    $(wildcard include/config/RWSEM_SPIN_ON_OWNER) \
    $(wildcard include/config/DEBUG_RWSEMS) \
  /its/home/mm2350/Desktop/bbr/include/linux/completion.h \
  /its/home/mm2350/Desktop/bbr/include/linux/swait.h \
  /its/home/mm2350/Desktop/bbr/include/linux/uprobes.h \
  /its/home/mm2350/Desktop/bbr/include/linux/timer.h \
    $(wildcard include/config/DEBUG_OBJECTS_TIMERS) \
  /its/home/mm2350/Desktop/bbr/include/linux/ktime.h \
  /its/home/mm2350/Desktop/bbr/include/linux/jiffies.h \
  /its/home/mm2350/Desktop/bbr/include/vdso/jiffies.h \
  /its/home/mm2350/Desktop/bbr/include/generated/timeconst.h \
  /its/home/mm2350/Desktop/bbr/include/vdso/ktime.h \
  /its/home/mm2350/Desktop/bbr/include/linux/timekeeping.h \
    $(wildcard include/config/GENERIC_CMOS_UPDATE) \
  /its/home/mm2350/Desktop/bbr/include/linux/clocksource_ids.h \
  /its/home/mm2350/Desktop/bbr/include/linux/debugobjects.h \
    $(wildcard include/config/DEBUG_OBJECTS) \
    $(wildcard include/config/DEBUG_OBJECTS_FREE) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/uprobes.h \
  /its/home/mm2350/Desktop/bbr/include/linux/notifier.h \
  /its/home/mm2350/Desktop/bbr/include/linux/srcu.h \
    $(wildcard include/config/TINY_SRCU) \
    $(wildcard include/config/NEED_SRCU_NMI_SAFE) \
  /its/home/mm2350/Desktop/bbr/include/linux/workqueue.h \
    $(wildcard include/config/DEBUG_OBJECTS_WORK) \
    $(wildcard include/config/FREEZER) \
    $(wildcard include/config/WQ_WATCHDOG) \
  /its/home/mm2350/Desktop/bbr/include/linux/workqueue_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rcu_segcblist.h \
  /its/home/mm2350/Desktop/bbr/include/linux/srcutree.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rcu_node_tree.h \
    $(wildcard include/config/RCU_FANOUT) \
    $(wildcard include/config/RCU_FANOUT_LEAF) \
  /its/home/mm2350/Desktop/bbr/include/linux/percpu_counter.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/mmu.h \
    $(wildcard include/config/MODIFY_LDT_SYSCALL) \
  /its/home/mm2350/Desktop/bbr/include/linux/page-flags.h \
    $(wildcard include/config/PAGE_IDLE_FLAG) \
    $(wildcard include/config/ARCH_USES_PG_ARCH_2) \
    $(wildcard include/config/ARCH_USES_PG_ARCH_3) \
    $(wildcard include/config/HUGETLB_PAGE_OPTIMIZE_VMEMMAP) \
  /its/home/mm2350/Desktop/bbr/include/linux/local_lock.h \
  /its/home/mm2350/Desktop/bbr/include/linux/local_lock_internal.h \
  /its/home/mm2350/Desktop/bbr/include/linux/zswap.h \
    $(wildcard include/config/ZSWAP) \
  /its/home/mm2350/Desktop/bbr/include/linux/memory_hotplug.h \
    $(wildcard include/config/ARCH_HAS_ADD_PAGES) \
    $(wildcard include/config/MEMORY_HOTREMOVE) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/asm/mmzone.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/mmzone.h \
  /its/home/mm2350/Desktop/bbr/include/linux/topology.h \
    $(wildcard include/config/USE_PERCPU_NUMA_NODE_ID) \
    $(wildcard include/config/SCHED_SMT) \
  /its/home/mm2350/Desktop/bbr/include/linux/arch_topology.h \
    $(wildcard include/config/GENERIC_ARCH_TOPOLOGY) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/topology.h \
    $(wildcard include/config/X86_LOCAL_APIC) \
    $(wildcard include/config/SCHED_MC_PRIO) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/mpspec.h \
    $(wildcard include/config/EISA) \
    $(wildcard include/config/X86_MPPARSE) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/mpspec_def.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/x86_init.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/apicdef.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/topology.h \
  /its/home/mm2350/Desktop/bbr/include/linux/cpu_smt.h \
    $(wildcard include/config/HOTPLUG_SMT) \
  /its/home/mm2350/Desktop/bbr/include/linux/sysctl.h \
    $(wildcard include/config/SYSCTL) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/sysctl.h \
  /its/home/mm2350/Desktop/bbr/include/linux/elf.h \
    $(wildcard include/config/ARCH_HAVE_EXTRA_ELF_NOTES) \
    $(wildcard include/config/ARCH_USE_GNU_PROPERTY) \
    $(wildcard include/config/ARCH_HAVE_ELF_PROT) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/elf.h \
    $(wildcard include/config/X86_X32_ABI) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/ia32.h \
  /its/home/mm2350/Desktop/bbr/include/linux/compat.h \
    $(wildcard include/config/ARCH_HAS_SYSCALL_WRAPPER) \
    $(wildcard include/config/COMPAT_OLD_SIGACTION) \
    $(wildcard include/config/ODD_RT_SIGACTION) \
  /its/home/mm2350/Desktop/bbr/include/linux/sem.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/sem.h \
  /its/home/mm2350/Desktop/bbr/include/linux/ipc.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rhashtable-types.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/ipc.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/ipcbuf.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/ipcbuf.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/sembuf.h \
  /its/home/mm2350/Desktop/bbr/include/linux/socket.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/socket.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/socket.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/sockios.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/sockios.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/sockios.h \
  /its/home/mm2350/Desktop/bbr/include/linux/uio.h \
    $(wildcard include/config/ARCH_HAS_COPY_MC) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/uio.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/socket.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/if.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/libc-compat.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/hdlc/ioctl.h \
  /its/home/mm2350/Desktop/bbr/include/linux/fs.h \
    $(wildcard include/config/READ_ONLY_THP_FOR_FS) \
    $(wildcard include/config/FS_POSIX_ACL) \
    $(wildcard include/config/CGROUP_WRITEBACK) \
    $(wildcard include/config/IMA) \
    $(wildcard include/config/FILE_LOCKING) \
    $(wildcard include/config/FSNOTIFY) \
    $(wildcard include/config/FS_ENCRYPTION) \
    $(wildcard include/config/FS_VERITY) \
    $(wildcard include/config/EPOLL) \
    $(wildcard include/config/UNICODE) \
    $(wildcard include/config/QUOTA) \
    $(wildcard include/config/FS_DAX) \
    $(wildcard include/config/BLOCK) \
  /its/home/mm2350/Desktop/bbr/include/linux/wait_bit.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kdev_t.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/kdev_t.h \
  /its/home/mm2350/Desktop/bbr/include/linux/dcache.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rculist.h \
    $(wildcard include/config/PROVE_RCU_LIST) \
  /its/home/mm2350/Desktop/bbr/include/linux/rculist_bl.h \
  /its/home/mm2350/Desktop/bbr/include/linux/list_bl.h \
  /its/home/mm2350/Desktop/bbr/include/linux/bit_spinlock.h \
  /its/home/mm2350/Desktop/bbr/include/linux/lockref.h \
    $(wildcard include/config/ARCH_USE_CMPXCHG_LOCKREF) \
  /its/home/mm2350/Desktop/bbr/include/linux/stringhash.h \
    $(wildcard include/config/DCACHE_WORD_ACCESS) \
  /its/home/mm2350/Desktop/bbr/include/linux/hash.h \
    $(wildcard include/config/HAVE_ARCH_HASH) \
  /its/home/mm2350/Desktop/bbr/include/linux/path.h \
  /its/home/mm2350/Desktop/bbr/include/linux/list_lru.h \
  /its/home/mm2350/Desktop/bbr/include/linux/shrinker.h \
    $(wildcard include/config/SHRINKER_DEBUG) \
  /its/home/mm2350/Desktop/bbr/include/linux/xarray.h \
    $(wildcard include/config/XARRAY_MULTI) \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/mm.h \
    $(wildcard include/config/MMU_LAZY_TLB_REFCOUNT) \
    $(wildcard include/config/ARCH_HAS_MEMBARRIER_CALLBACKS) \
  /its/home/mm2350/Desktop/bbr/include/linux/sync_core.h \
    $(wildcard include/config/ARCH_HAS_SYNC_CORE_BEFORE_USERMODE) \
    $(wildcard include/config/ARCH_HAS_PREPARE_SYNC_CORE_CMD) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/sync_core.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/coredump.h \
  /its/home/mm2350/Desktop/bbr/include/linux/radix-tree.h \
  /its/home/mm2350/Desktop/bbr/include/linux/pid.h \
  /its/home/mm2350/Desktop/bbr/include/linux/capability.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/capability.h \
  /its/home/mm2350/Desktop/bbr/include/linux/semaphore.h \
  /its/home/mm2350/Desktop/bbr/include/linux/fcntl.h \
    $(wildcard include/config/ARCH_32BIT_OFF_T) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/fcntl.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/fcntl.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/asm-generic/fcntl.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/openat2.h \
  /its/home/mm2350/Desktop/bbr/include/linux/migrate_mode.h \
  /its/home/mm2350/Desktop/bbr/include/linux/percpu-rwsem.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rcuwait.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/signal.h \
    $(wildcard include/config/SCHED_AUTOGROUP) \
    $(wildcard include/config/BSD_PROCESS_ACCT) \
    $(wildcard include/config/TASKSTATS) \
    $(wildcard include/config/STACK_GROWSUP) \
  /its/home/mm2350/Desktop/bbr/include/linux/signal.h \
    $(wildcard include/config/DYNAMIC_SIGFRAME) \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/jobctl.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/task.h \
    $(wildcard include/config/HAVE_EXIT_THREAD) \
    $(wildcard include/config/ARCH_WANTS_DYNAMIC_TASK_STRUCT) \
    $(wildcard include/config/HAVE_ARCH_THREAD_STRUCT_WHITELIST) \
  /its/home/mm2350/Desktop/bbr/include/linux/uaccess.h \
    $(wildcard include/config/ARCH_HAS_SUBPAGE_FAULTS) \
  /its/home/mm2350/Desktop/bbr/include/linux/fault-inject-usercopy.h \
    $(wildcard include/config/FAULT_INJECTION_USERCOPY) \
  /its/home/mm2350/Desktop/bbr/include/linux/nospec.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/uaccess.h \
    $(wildcard include/config/CC_HAS_ASM_GOTO_OUTPUT) \
    $(wildcard include/config/CC_HAS_ASM_GOTO_TIED_OUTPUT) \
    $(wildcard include/config/X86_INTEL_USERCOPY) \
  /its/home/mm2350/Desktop/bbr/include/linux/mmap_lock.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/smap.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/extable.h \
    $(wildcard include/config/BPF_JIT) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/tlbflush.h \
  /its/home/mm2350/Desktop/bbr/include/linux/mmu_notifier.h \
  /its/home/mm2350/Desktop/bbr/include/linux/interval_tree.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/invpcid.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/pti.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/pgtable.h \
    $(wildcard include/config/DEBUG_WX) \
    $(wildcard include/config/HAVE_ARCH_TRANSPARENT_HUGEPAGE_PUD) \
    $(wildcard include/config/ARCH_HAS_PTE_DEVMAP) \
    $(wildcard include/config/ARCH_SUPPORTS_PMD_PFNMAP) \
    $(wildcard include/config/ARCH_SUPPORTS_PUD_PFNMAP) \
    $(wildcard include/config/HAVE_ARCH_SOFT_DIRTY) \
    $(wildcard include/config/ARCH_ENABLE_THP_MIGRATION) \
    $(wildcard include/config/PAGE_TABLE_CHECK) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/pkru.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/fpu/api.h \
    $(wildcard include/config/X86_DEBUG_FPU) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/coco.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/pgtable_uffd.h \
  /its/home/mm2350/Desktop/bbr/include/linux/page_table_check.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/pgtable_64.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/fixmap.h \
    $(wildcard include/config/PROVIDE_OHCI1394_DMA_INIT) \
    $(wildcard include/config/X86_IO_APIC) \
    $(wildcard include/config/PCI_MMCONFIG) \
    $(wildcard include/config/ACPI_APEI_GHES) \
    $(wildcard include/config/INTEL_TXT) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/vsyscall.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/fixmap.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/pgtable-invert.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/uaccess_64.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/runtime-const.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/access_ok.h \
    $(wildcard include/config/ALTERNATE_USER_ADDRESS_SPACE) \
  /its/home/mm2350/Desktop/bbr/include/linux/cred.h \
  /its/home/mm2350/Desktop/bbr/include/linux/key.h \
    $(wildcard include/config/KEY_NOTIFICATIONS) \
    $(wildcard include/config/NET) \
  /its/home/mm2350/Desktop/bbr/include/linux/assoc_array.h \
    $(wildcard include/config/ASSOCIATIVE_ARRAY) \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/user.h \
    $(wildcard include/config/VFIO_PCI_ZDEV_KVM) \
    $(wildcard include/config/IOMMUFD) \
    $(wildcard include/config/WATCH_QUEUE) \
  /its/home/mm2350/Desktop/bbr/include/linux/ratelimit.h \
  /its/home/mm2350/Desktop/bbr/include/linux/posix-timers.h \
  /its/home/mm2350/Desktop/bbr/include/linux/alarmtimer.h \
    $(wildcard include/config/RTC_CLASS) \
  /its/home/mm2350/Desktop/bbr/include/linux/hrtimer.h \
    $(wildcard include/config/HIGH_RES_TIMERS) \
    $(wildcard include/config/TIME_LOW_RES) \
    $(wildcard include/config/TIMERFD) \
  /its/home/mm2350/Desktop/bbr/include/linux/hrtimer_defs.h \
  /its/home/mm2350/Desktop/bbr/include/linux/timerqueue.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rcuref.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rcu_sync.h \
  /its/home/mm2350/Desktop/bbr/include/linux/delayed_call.h \
  /its/home/mm2350/Desktop/bbr/include/linux/uuid.h \
  /its/home/mm2350/Desktop/bbr/include/linux/errseq.h \
  /its/home/mm2350/Desktop/bbr/include/linux/ioprio.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/rt.h \
  /its/home/mm2350/Desktop/bbr/include/linux/iocontext.h \
    $(wildcard include/config/BLK_ICQ) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/ioprio.h \
  /its/home/mm2350/Desktop/bbr/include/linux/fs_types.h \
  /its/home/mm2350/Desktop/bbr/include/linux/mount.h \
  /its/home/mm2350/Desktop/bbr/include/linux/mnt_idmapping.h \
  /its/home/mm2350/Desktop/bbr/include/linux/slab.h \
    $(wildcard include/config/FAILSLAB) \
    $(wildcard include/config/KFENCE) \
    $(wildcard include/config/SLUB_TINY) \
    $(wildcard include/config/SLUB_DEBUG) \
    $(wildcard include/config/SLAB_FREELIST_HARDENED) \
    $(wildcard include/config/SLAB_BUCKETS) \
  /its/home/mm2350/Desktop/bbr/include/linux/percpu-refcount.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kasan.h \
    $(wildcard include/config/KASAN_STACK) \
    $(wildcard include/config/KASAN_VMALLOC) \
  /its/home/mm2350/Desktop/bbr/include/linux/kasan-enabled.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kasan-tags.h \
  /its/home/mm2350/Desktop/bbr/include/linux/rw_hint.h \
  /its/home/mm2350/Desktop/bbr/include/linux/file_ref.h \
  /its/home/mm2350/Desktop/bbr/include/linux/unicode.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/fs.h \
  /its/home/mm2350/Desktop/bbr/include/linux/quota.h \
    $(wildcard include/config/QUOTA_NETLINK_INTERFACE) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/dqblk_xfs.h \
  /its/home/mm2350/Desktop/bbr/include/linux/dqblk_v1.h \
  /its/home/mm2350/Desktop/bbr/include/linux/dqblk_v2.h \
  /its/home/mm2350/Desktop/bbr/include/linux/dqblk_qtree.h \
  /its/home/mm2350/Desktop/bbr/include/linux/projid.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/quota.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/aio_abi.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/unistd.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/unistd.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/uapi/asm/unistd.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/uapi/asm/unistd_64.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/asm/unistd_64_x32.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/generated/asm/unistd_32_ia32.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/compat.h \
  /its/home/mm2350/Desktop/bbr/include/linux/sched/task_stack.h \
    $(wildcard include/config/DEBUG_STACK_USAGE) \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/magic.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/user32.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/compat.h \
    $(wildcard include/config/COMPAT_FOR_U64_ALIGNMENT) \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/syscall_wrapper.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/user.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/user_64.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/fsgsbase.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/vdso.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/elf.h \
  /its/home/mm2350/Desktop/bbr/include/uapi/linux/elf-em.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kobject.h \
    $(wildcard include/config/UEVENT_HELPER) \
    $(wildcard include/config/DEBUG_KOBJECT_RELEASE) \
  /its/home/mm2350/Desktop/bbr/include/linux/sysfs.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kernfs.h \
    $(wildcard include/config/KERNFS) \
  /its/home/mm2350/Desktop/bbr/include/linux/idr.h \
  /its/home/mm2350/Desktop/bbr/include/linux/kobject_ns.h \
  /its/home/mm2350/Desktop/bbr/include/linux/moduleparam.h \
    $(wildcard include/config/ALPHA) \
    $(wildcard include/config/PPC64) \
  /its/home/mm2350/Desktop/bbr/include/linux/rbtree_latch.h \
  /its/home/mm2350/Desktop/bbr/include/linux/error-injection.h \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/error-injection.h \
  /its/home/mm2350/Desktop/bbr/arch/x86/include/asm/module.h \
    $(wildcard include/config/UNWINDER_ORC) \
  /its/home/mm2350/Desktop/bbr/include/asm-generic/module.h \
    $(wildcard include/config/HAVE_MOD_ARCH_SPECIFIC) \
    $(wildcard include/config/MODULES_USE_ELF_REL) \
    $(wildcard include/config/MODULES_USE_ELF_RELA) \
  /its/home/mm2350/Desktop/bbr/include/linux/export-internal.h \
    $(wildcard include/config/PARISC) \

mutant.mod.o: $(deps_mutant.mod.o)

$(deps_mutant.mod.o):

mutant.mod.o: $(wildcard /its/home/mm2350/Desktop/bbr/tools/objtool/objtool)
