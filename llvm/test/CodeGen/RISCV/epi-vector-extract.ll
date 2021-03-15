; RUN: llc -mtriple=riscv64 -mattr=+experimental-v,+f,+d -target-abi lp64d \
; RUN:    -verify-machineinstrs < %s -epi-pipeline -o /dev/null

define i1 @extract_nxv1i1(<vscale x 1 x i1> %v, i64 %idx)
{
  %elem = extractelement <vscale x 1 x i1> %v, i64 %idx
  %first = extractelement <vscale x 1 x i1> %v, i64 0
  %res = add i1 %elem, %first
  ret i1 %res
}

define i1 @extract_nxv2i1(<vscale x 2 x i1> %v, i64 %idx)
{
  %elem = extractelement <vscale x 2 x i1> %v, i64 %idx
  %first = extractelement <vscale x 2 x i1> %v, i64 0
  %res = add i1 %elem, %first
  ret i1 %res
}

define i1 @extract_nxv4i1(<vscale x 4 x i1> %v, i64 %idx)
{
  %elem = extractelement <vscale x 4 x i1> %v, i64 %idx
  %first = extractelement <vscale x 4 x i1> %v, i64 0
  %res = add i1 %elem, %first
  ret i1 %res
}

define i1 @extract_nxv8i1(<vscale x 8 x i1> %v, i64 %idx)
{
  %elem = extractelement <vscale x 8 x i1> %v, i64 %idx
  %first = extractelement <vscale x 8 x i1> %v, i64 0
  %res = add i1 %elem, %first
  ret i1 %res
}

define i1 @extract_nxv16i1(<vscale x 16 x i1> %v, i64 %idx)
{
  %elem = extractelement <vscale x 16 x i1> %v, i64 %idx
  %first = extractelement <vscale x 16 x i1> %v, i64 0
  %res = add i1 %elem, %first
  ret i1 %res
}

define i1 @extract_nxv32i1(<vscale x 32 x i1> %v, i64 %idx)
{
  %elem = extractelement <vscale x 32 x i1> %v, i64 %idx
  %first = extractelement <vscale x 32 x i1> %v, i64 0
  %res = add i1 %elem, %first
  ret i1 %res
}

define i1 @extract_nxv64i1(<vscale x 64 x i1> %v, i64 %idx)
{
  %elem = extractelement <vscale x 64 x i1> %v, i64 %idx
  %first = extractelement <vscale x 64 x i1> %v, i64 0
  %res = add i1 %elem, %first
  ret i1 %res
}
