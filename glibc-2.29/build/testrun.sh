#!/bin/bash
builddir=`dirname "$0"`
GCONV_PATH="${builddir}/iconvdata"

usage () {
  echo "usage: $0 [--tool=strace] PROGRAM [ARGUMENTS...]" 2>&1
  echo "       $0 --tool=valgrind PROGRAM [ARGUMENTS...]" 2>&1
}

toolname=default
while test $# -gt 0 ; do
  case "$1" in
    --tool=*)
      toolname="${1:7}"
      shift
      ;;
    --*)
      usage
      ;;
    *)
      break
      ;;
  esac
done

if test $# -eq 0 ; then
  usage
fi

case "$toolname" in
  default)
    exec   env GCONV_PATH="${builddir}"/iconvdata LOCPATH="${builddir}"/localedata LC_ALL=C  "${builddir}"/elf/ld-linux-x86-64.so.2 --library-path "${builddir}":"${builddir}"/math:"${builddir}"/elf:"${builddir}"/dlfcn:"${builddir}"/nss:"${builddir}"/nis:"${builddir}"/rt:"${builddir}"/resolv:"${builddir}"/mathvec:"${builddir}"/support:"${builddir}"/crypt:"${builddir}"/nptl ${1+"$@"}
    ;;
  strace)
    exec strace  -EGCONV_PATH=/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/iconvdata  -ELOCPATH=/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/localedata  -ELC_ALL=C  /scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/elf/ld-linux-x86-64.so.2 --library-path /scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/math:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/elf:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/dlfcn:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/nss:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/nis:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/rt:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/resolv:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/mathvec:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/support:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/crypt:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/nptl ${1+"$@"}
    ;;
  valgrind)
    exec env GCONV_PATH=/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/iconvdata LOCPATH=/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/localedata LC_ALL=C valgrind  /scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/elf/ld-linux-x86-64.so.2 --library-path /scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/math:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/elf:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/dlfcn:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/nss:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/nis:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/rt:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/resolv:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/mathvec:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/support:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/crypt:/scratch/ca2627/clinicalAI/ArabicLLMBench/glibc-2.29/build/nptl ${1+"$@"}
    ;;
  *)
    usage
    ;;
esac
