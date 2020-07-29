; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=tahiti < %s | FileCheck -check-prefixes=GCN,GFX6 %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=fiji < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,GFX9 %s

define i8 @v_saddsat_i8(i8 %lhs, i8 %rhs) {
; GFX6-LABEL: v_saddsat_i8:
; GFX6:       ; %bb.0:
; GFX6-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX6-NEXT:    v_bfe_i32 v1, v1, 0, 8
; GFX6-NEXT:    v_bfe_i32 v0, v0, 0, 8
; GFX6-NEXT:    v_add_i32_e32 v0, vcc, v0, v1
; GFX6-NEXT:    v_min_i32_e32 v0, 0x7f, v0
; GFX6-NEXT:    v_max_i32_e32 v0, 0xffffff80, v0
; GFX6-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: v_saddsat_i8:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    v_add_u16_sdwa v0, sext(v0), sext(v1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0
; GFX8-NEXT:    v_min_i16_e32 v0, 0x7f, v0
; GFX8-NEXT:    v_max_i16_e32 v0, 0xff80, v0
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_saddsat_i8:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_lshlrev_b16_e32 v1, 8, v1
; GFX9-NEXT:    v_lshlrev_b16_e32 v0, 8, v0
; GFX9-NEXT:    v_add_i16 v0, v0, v1 clamp
; GFX9-NEXT:    v_ashrrev_i16_e32 v0, 8, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call i8 @llvm.sadd.sat.i8(i8 %lhs, i8 %rhs)
  ret i8 %result
}

define i16 @v_saddsat_i16(i16 %lhs, i16 %rhs) {
; GFX6-LABEL: v_saddsat_i16:
; GFX6:       ; %bb.0:
; GFX6-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX6-NEXT:    v_bfe_i32 v1, v1, 0, 16
; GFX6-NEXT:    v_bfe_i32 v0, v0, 0, 16
; GFX6-NEXT:    v_add_i32_e32 v0, vcc, v0, v1
; GFX6-NEXT:    v_min_i32_e32 v0, 0x7fff, v0
; GFX6-NEXT:    v_max_i32_e32 v0, 0xffff8000, v0
; GFX6-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: v_saddsat_i16:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v1
; GFX8-NEXT:    v_add_u16_e32 v1, v0, v1
; GFX8-NEXT:    v_cmp_lt_i16_e64 s[4:5], v1, v0
; GFX8-NEXT:    v_mov_b32_e32 v0, 0xffff8000
; GFX8-NEXT:    v_mov_b32_e32 v2, 0x7fff
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[6:7], 0, v1
; GFX8-NEXT:    v_cndmask_b32_e64 v0, v0, v2, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_cndmask_b32_e32 v0, v1, v0, vcc
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_saddsat_i16:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_add_i16 v0, v0, v1 clamp
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call i16 @llvm.sadd.sat.i16(i16 %lhs, i16 %rhs)
  ret i16 %result
}

define i32 @v_saddsat_i32(i32 %lhs, i32 %rhs) {
; GFX6-LABEL: v_saddsat_i32:
; GFX6:       ; %bb.0:
; GFX6-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX6-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v1
; GFX6-NEXT:    v_add_i32_e64 v1, s[4:5], v0, v1
; GFX6-NEXT:    v_cmp_lt_i32_e64 s[4:5], v1, v0
; GFX6-NEXT:    v_bfrev_b32_e32 v0, 1
; GFX6-NEXT:    v_bfrev_b32_e32 v2, -2
; GFX6-NEXT:    v_cmp_gt_i32_e64 s[6:7], 0, v1
; GFX6-NEXT:    v_cndmask_b32_e64 v0, v0, v2, s[6:7]
; GFX6-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX6-NEXT:    v_cndmask_b32_e32 v0, v1, v0, vcc
; GFX6-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: v_saddsat_i32:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v1
; GFX8-NEXT:    v_add_u32_e64 v1, s[4:5], v0, v1
; GFX8-NEXT:    v_cmp_lt_i32_e64 s[4:5], v1, v0
; GFX8-NEXT:    v_bfrev_b32_e32 v0, 1
; GFX8-NEXT:    v_bfrev_b32_e32 v2, -2
; GFX8-NEXT:    v_cmp_gt_i32_e64 s[6:7], 0, v1
; GFX8-NEXT:    v_cndmask_b32_e64 v0, v0, v2, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_cndmask_b32_e32 v0, v1, v0, vcc
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_saddsat_i32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_add_i32 v0, v0, v1 clamp
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call i32 @llvm.sadd.sat.i32(i32 %lhs, i32 %rhs)
  ret i32 %result
}

define <2 x i16> @v_saddsat_v2i16(<2 x i16> %lhs, <2 x i16> %rhs) {
; GFX6-LABEL: v_saddsat_v2i16:
; GFX6:       ; %bb.0:
; GFX6-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX6-NEXT:    v_bfe_i32 v2, v2, 0, 16
; GFX6-NEXT:    v_bfe_i32 v0, v0, 0, 16
; GFX6-NEXT:    v_bfe_i32 v3, v3, 0, 16
; GFX6-NEXT:    v_bfe_i32 v1, v1, 0, 16
; GFX6-NEXT:    v_add_i32_e32 v1, vcc, v1, v3
; GFX6-NEXT:    s_movk_i32 s4, 0x7fff
; GFX6-NEXT:    v_add_i32_e32 v0, vcc, v0, v2
; GFX6-NEXT:    v_min_i32_e32 v1, s4, v1
; GFX6-NEXT:    s_movk_i32 s5, 0x8000
; GFX6-NEXT:    v_min_i32_e32 v0, s4, v0
; GFX6-NEXT:    v_max_i32_e32 v1, s5, v1
; GFX6-NEXT:    v_max_i32_e32 v0, s5, v0
; GFX6-NEXT:    v_lshlrev_b32_e32 v1, 16, v1
; GFX6-NEXT:    v_and_b32_e32 v0, 0xffff, v0
; GFX6-NEXT:    v_or_b32_e32 v0, v0, v1
; GFX6-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX6-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: v_saddsat_v2i16:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    v_lshrrev_b32_e32 v2, 16, v1
; GFX8-NEXT:    v_lshrrev_b32_e32 v3, 16, v0
; GFX8-NEXT:    v_add_u16_e32 v4, v3, v2
; GFX8-NEXT:    v_mov_b32_e32 v5, 0xffff8000
; GFX8-NEXT:    v_mov_b32_e32 v6, 0x7fff
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v4
; GFX8-NEXT:    v_cndmask_b32_e32 v7, v5, v6, vcc
; GFX8-NEXT:    v_cmp_lt_i16_e32 vcc, v4, v3
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[4:5], 0, v2
; GFX8-NEXT:    s_xor_b64 vcc, s[4:5], vcc
; GFX8-NEXT:    v_cndmask_b32_e32 v2, v4, v7, vcc
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v1
; GFX8-NEXT:    v_add_u16_e32 v1, v0, v1
; GFX8-NEXT:    v_cmp_lt_i16_e64 s[4:5], v1, v0
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[6:7], 0, v1
; GFX8-NEXT:    v_cndmask_b32_e64 v0, v5, v6, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_lshlrev_b32_e32 v2, 16, v2
; GFX8-NEXT:    v_cndmask_b32_e32 v0, v1, v0, vcc
; GFX8-NEXT:    v_or_b32_sdwa v0, v0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_saddsat_v2i16:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_pk_add_i16 v0, v0, v1 clamp
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call <2 x i16> @llvm.sadd.sat.v2i16(<2 x i16> %lhs, <2 x i16> %rhs)
  ret <2 x i16> %result
}

define <3 x i16> @v_saddsat_v3i16(<3 x i16> %lhs, <3 x i16> %rhs) {
; GFX6-LABEL: v_saddsat_v3i16:
; GFX6:       ; %bb.0:
; GFX6-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX6-NEXT:    v_bfe_i32 v3, v3, 0, 16
; GFX6-NEXT:    v_bfe_i32 v0, v0, 0, 16
; GFX6-NEXT:    v_bfe_i32 v4, v4, 0, 16
; GFX6-NEXT:    v_bfe_i32 v1, v1, 0, 16
; GFX6-NEXT:    v_add_i32_e32 v1, vcc, v1, v4
; GFX6-NEXT:    s_movk_i32 s4, 0x7fff
; GFX6-NEXT:    v_add_i32_e32 v0, vcc, v0, v3
; GFX6-NEXT:    v_min_i32_e32 v1, s4, v1
; GFX6-NEXT:    s_movk_i32 s5, 0x8000
; GFX6-NEXT:    v_min_i32_e32 v0, s4, v0
; GFX6-NEXT:    v_max_i32_e32 v1, s5, v1
; GFX6-NEXT:    v_max_i32_e32 v0, s5, v0
; GFX6-NEXT:    v_bfe_i32 v5, v5, 0, 16
; GFX6-NEXT:    v_bfe_i32 v2, v2, 0, 16
; GFX6-NEXT:    v_lshlrev_b32_e32 v1, 16, v1
; GFX6-NEXT:    v_and_b32_e32 v0, 0xffff, v0
; GFX6-NEXT:    v_or_b32_e32 v0, v0, v1
; GFX6-NEXT:    v_add_i32_e32 v1, vcc, v2, v5
; GFX6-NEXT:    v_min_i32_e32 v1, s4, v1
; GFX6-NEXT:    v_max_i32_e32 v1, s5, v1
; GFX6-NEXT:    v_or_b32_e32 v2, 0xffff0000, v1
; GFX6-NEXT:    v_alignbit_b32 v1, v1, v0, 16
; GFX6-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: v_saddsat_v3i16:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    v_lshrrev_b32_e32 v4, 16, v2
; GFX8-NEXT:    v_lshrrev_b32_e32 v5, 16, v0
; GFX8-NEXT:    v_add_u16_e32 v6, v5, v4
; GFX8-NEXT:    v_mov_b32_e32 v7, 0xffff8000
; GFX8-NEXT:    v_mov_b32_e32 v8, 0x7fff
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v6
; GFX8-NEXT:    v_cndmask_b32_e32 v9, v7, v8, vcc
; GFX8-NEXT:    v_cmp_lt_i16_e32 vcc, v6, v5
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[4:5], 0, v4
; GFX8-NEXT:    s_xor_b64 vcc, s[4:5], vcc
; GFX8-NEXT:    v_cndmask_b32_e32 v4, v6, v9, vcc
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v3
; GFX8-NEXT:    v_add_u16_e32 v3, v1, v3
; GFX8-NEXT:    v_cmp_lt_i16_e64 s[4:5], v3, v1
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[6:7], 0, v3
; GFX8-NEXT:    v_cndmask_b32_e64 v1, v7, v8, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_cndmask_b32_e32 v1, v3, v1, vcc
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v2
; GFX8-NEXT:    v_add_u16_e32 v2, v0, v2
; GFX8-NEXT:    v_cmp_lt_i16_e64 s[4:5], v2, v0
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[6:7], 0, v2
; GFX8-NEXT:    v_cndmask_b32_e64 v0, v7, v8, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_cndmask_b32_e32 v0, v2, v0, vcc
; GFX8-NEXT:    v_lshlrev_b32_e32 v2, 16, v4
; GFX8-NEXT:    v_or_b32_sdwa v0, v0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_saddsat_v3i16:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_pk_add_i16 v1, v1, v3 clamp
; GFX9-NEXT:    v_pk_add_i16 v0, v0, v2 clamp
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call <3 x i16> @llvm.sadd.sat.v3i16(<3 x i16> %lhs, <3 x i16> %rhs)
  ret <3 x i16> %result
}

define <2 x float> @v_saddsat_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; GFX6-LABEL: v_saddsat_v4i16:
; GFX6:       ; %bb.0:
; GFX6-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX6-NEXT:    v_bfe_i32 v4, v4, 0, 16
; GFX6-NEXT:    v_bfe_i32 v0, v0, 0, 16
; GFX6-NEXT:    v_bfe_i32 v5, v5, 0, 16
; GFX6-NEXT:    v_bfe_i32 v1, v1, 0, 16
; GFX6-NEXT:    v_add_i32_e32 v1, vcc, v1, v5
; GFX6-NEXT:    s_movk_i32 s4, 0x7fff
; GFX6-NEXT:    v_add_i32_e32 v0, vcc, v0, v4
; GFX6-NEXT:    v_min_i32_e32 v1, s4, v1
; GFX6-NEXT:    s_movk_i32 s5, 0x8000
; GFX6-NEXT:    v_min_i32_e32 v0, s4, v0
; GFX6-NEXT:    v_max_i32_e32 v1, s5, v1
; GFX6-NEXT:    v_max_i32_e32 v0, s5, v0
; GFX6-NEXT:    s_mov_b32 s6, 0xffff
; GFX6-NEXT:    v_bfe_i32 v6, v6, 0, 16
; GFX6-NEXT:    v_bfe_i32 v2, v2, 0, 16
; GFX6-NEXT:    v_bfe_i32 v7, v7, 0, 16
; GFX6-NEXT:    v_bfe_i32 v3, v3, 0, 16
; GFX6-NEXT:    v_lshlrev_b32_e32 v1, 16, v1
; GFX6-NEXT:    v_and_b32_e32 v0, s6, v0
; GFX6-NEXT:    v_add_i32_e32 v2, vcc, v2, v6
; GFX6-NEXT:    v_or_b32_e32 v0, v0, v1
; GFX6-NEXT:    v_add_i32_e32 v1, vcc, v3, v7
; GFX6-NEXT:    v_min_i32_e32 v1, s4, v1
; GFX6-NEXT:    v_min_i32_e32 v2, s4, v2
; GFX6-NEXT:    v_max_i32_e32 v1, s5, v1
; GFX6-NEXT:    v_max_i32_e32 v2, s5, v2
; GFX6-NEXT:    v_lshlrev_b32_e32 v1, 16, v1
; GFX6-NEXT:    v_and_b32_e32 v2, s6, v2
; GFX6-NEXT:    v_or_b32_e32 v1, v2, v1
; GFX6-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: v_saddsat_v4i16:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    v_lshrrev_b32_e32 v4, 16, v2
; GFX8-NEXT:    v_lshrrev_b32_e32 v5, 16, v0
; GFX8-NEXT:    v_add_u16_e32 v6, v5, v4
; GFX8-NEXT:    v_mov_b32_e32 v7, 0xffff8000
; GFX8-NEXT:    v_mov_b32_e32 v8, 0x7fff
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v6
; GFX8-NEXT:    v_cndmask_b32_e32 v9, v7, v8, vcc
; GFX8-NEXT:    v_cmp_lt_i16_e32 vcc, v6, v5
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[4:5], 0, v4
; GFX8-NEXT:    s_xor_b64 vcc, s[4:5], vcc
; GFX8-NEXT:    v_cndmask_b32_e32 v4, v6, v9, vcc
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v2
; GFX8-NEXT:    v_add_u16_e32 v2, v0, v2
; GFX8-NEXT:    v_cmp_lt_i16_e64 s[4:5], v2, v0
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[6:7], 0, v2
; GFX8-NEXT:    v_cndmask_b32_e64 v0, v7, v8, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_cndmask_b32_e32 v0, v2, v0, vcc
; GFX8-NEXT:    v_lshlrev_b32_e32 v4, 16, v4
; GFX8-NEXT:    v_or_b32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX8-NEXT:    v_lshrrev_b32_e32 v2, 16, v3
; GFX8-NEXT:    v_lshrrev_b32_e32 v4, 16, v1
; GFX8-NEXT:    v_add_u16_e32 v5, v4, v2
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v5
; GFX8-NEXT:    v_cndmask_b32_e32 v6, v7, v8, vcc
; GFX8-NEXT:    v_cmp_lt_i16_e32 vcc, v5, v4
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[4:5], 0, v2
; GFX8-NEXT:    s_xor_b64 vcc, s[4:5], vcc
; GFX8-NEXT:    v_cndmask_b32_e32 v2, v5, v6, vcc
; GFX8-NEXT:    v_cmp_gt_i16_e32 vcc, 0, v3
; GFX8-NEXT:    v_add_u16_e32 v3, v1, v3
; GFX8-NEXT:    v_cmp_lt_i16_e64 s[4:5], v3, v1
; GFX8-NEXT:    v_cmp_gt_i16_e64 s[6:7], 0, v3
; GFX8-NEXT:    v_cndmask_b32_e64 v1, v7, v8, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_lshlrev_b32_e32 v2, 16, v2
; GFX8-NEXT:    v_cndmask_b32_e32 v1, v3, v1, vcc
; GFX8-NEXT:    v_or_b32_sdwa v1, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_saddsat_v4i16:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_pk_add_i16 v0, v0, v2 clamp
; GFX9-NEXT:    v_pk_add_i16 v1, v1, v3 clamp
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
  %cast = bitcast <4 x i16> %result to <2 x float>
  ret <2 x float> %cast
}

define <2 x i32> @v_saddsat_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; GFX6-LABEL: v_saddsat_v2i32:
; GFX6:       ; %bb.0:
; GFX6-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX6-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v2
; GFX6-NEXT:    v_add_i32_e64 v2, s[4:5], v0, v2
; GFX6-NEXT:    v_cmp_lt_i32_e64 s[4:5], v2, v0
; GFX6-NEXT:    v_bfrev_b32_e32 v4, 1
; GFX6-NEXT:    v_bfrev_b32_e32 v5, -2
; GFX6-NEXT:    v_cmp_gt_i32_e64 s[6:7], 0, v2
; GFX6-NEXT:    v_cndmask_b32_e64 v0, v4, v5, s[6:7]
; GFX6-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX6-NEXT:    v_cndmask_b32_e32 v0, v2, v0, vcc
; GFX6-NEXT:    v_add_i32_e64 v2, s[4:5], v1, v3
; GFX6-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v3
; GFX6-NEXT:    v_cmp_lt_i32_e64 s[4:5], v2, v1
; GFX6-NEXT:    v_cmp_gt_i32_e64 s[6:7], 0, v2
; GFX6-NEXT:    v_cndmask_b32_e64 v1, v4, v5, s[6:7]
; GFX6-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX6-NEXT:    v_cndmask_b32_e32 v1, v2, v1, vcc
; GFX6-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: v_saddsat_v2i32:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v2
; GFX8-NEXT:    v_add_u32_e64 v2, s[4:5], v0, v2
; GFX8-NEXT:    v_cmp_lt_i32_e64 s[4:5], v2, v0
; GFX8-NEXT:    v_bfrev_b32_e32 v4, 1
; GFX8-NEXT:    v_bfrev_b32_e32 v5, -2
; GFX8-NEXT:    v_cmp_gt_i32_e64 s[6:7], 0, v2
; GFX8-NEXT:    v_cndmask_b32_e64 v0, v4, v5, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_cndmask_b32_e32 v0, v2, v0, vcc
; GFX8-NEXT:    v_add_u32_e64 v2, s[4:5], v1, v3
; GFX8-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v3
; GFX8-NEXT:    v_cmp_lt_i32_e64 s[4:5], v2, v1
; GFX8-NEXT:    v_cmp_gt_i32_e64 s[6:7], 0, v2
; GFX8-NEXT:    v_cndmask_b32_e64 v1, v4, v5, s[6:7]
; GFX8-NEXT:    s_xor_b64 vcc, vcc, s[4:5]
; GFX8-NEXT:    v_cndmask_b32_e32 v1, v2, v1, vcc
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_saddsat_v2i32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_add_i32 v0, v0, v2 clamp
; GFX9-NEXT:    v_add_i32 v1, v1, v3 clamp
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
  ret <2 x i32> %result
}

define i64 @v_saddsat_i64(i64 %lhs, i64 %rhs) {
; GFX6-LABEL: v_saddsat_i64:
; GFX6:       ; %bb.0:
; GFX6-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX6-NEXT:    v_add_i32_e32 v4, vcc, v0, v2
; GFX6-NEXT:    v_addc_u32_e32 v5, vcc, v1, v3, vcc
; GFX6-NEXT:    v_cmp_lt_i64_e32 vcc, v[4:5], v[0:1]
; GFX6-NEXT:    v_cmp_gt_i64_e64 s[4:5], 0, v[2:3]
; GFX6-NEXT:    v_bfrev_b32_e32 v1, 1
; GFX6-NEXT:    s_xor_b64 vcc, s[4:5], vcc
; GFX6-NEXT:    v_cmp_gt_i64_e64 s[4:5], 0, v[4:5]
; GFX6-NEXT:    v_bfrev_b32_e32 v2, -2
; GFX6-NEXT:    v_ashrrev_i32_e32 v0, 31, v5
; GFX6-NEXT:    v_cndmask_b32_e64 v1, v1, v2, s[4:5]
; GFX6-NEXT:    v_cndmask_b32_e32 v0, v4, v0, vcc
; GFX6-NEXT:    v_cndmask_b32_e32 v1, v5, v1, vcc
; GFX6-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: v_saddsat_i64:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    v_add_u32_e32 v4, vcc, v0, v2
; GFX8-NEXT:    v_addc_u32_e32 v5, vcc, v1, v3, vcc
; GFX8-NEXT:    v_cmp_lt_i64_e32 vcc, v[4:5], v[0:1]
; GFX8-NEXT:    v_cmp_gt_i64_e64 s[4:5], 0, v[2:3]
; GFX8-NEXT:    v_bfrev_b32_e32 v1, 1
; GFX8-NEXT:    s_xor_b64 vcc, s[4:5], vcc
; GFX8-NEXT:    v_cmp_gt_i64_e64 s[4:5], 0, v[4:5]
; GFX8-NEXT:    v_bfrev_b32_e32 v2, -2
; GFX8-NEXT:    v_ashrrev_i32_e32 v0, 31, v5
; GFX8-NEXT:    v_cndmask_b32_e64 v1, v1, v2, s[4:5]
; GFX8-NEXT:    v_cndmask_b32_e32 v0, v4, v0, vcc
; GFX8-NEXT:    v_cndmask_b32_e32 v1, v5, v1, vcc
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_saddsat_i64:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_add_co_u32_e32 v4, vcc, v0, v2
; GFX9-NEXT:    v_addc_co_u32_e32 v5, vcc, v1, v3, vcc
; GFX9-NEXT:    v_cmp_lt_i64_e32 vcc, v[4:5], v[0:1]
; GFX9-NEXT:    v_cmp_gt_i64_e64 s[4:5], 0, v[2:3]
; GFX9-NEXT:    v_bfrev_b32_e32 v1, 1
; GFX9-NEXT:    s_xor_b64 vcc, s[4:5], vcc
; GFX9-NEXT:    v_cmp_gt_i64_e64 s[4:5], 0, v[4:5]
; GFX9-NEXT:    v_bfrev_b32_e32 v2, -2
; GFX9-NEXT:    v_ashrrev_i32_e32 v0, 31, v5
; GFX9-NEXT:    v_cndmask_b32_e64 v1, v1, v2, s[4:5]
; GFX9-NEXT:    v_cndmask_b32_e32 v0, v4, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v1, v5, v1, vcc
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call i64 @llvm.sadd.sat.i64(i64 %lhs, i64 %rhs)
  ret i64 %result
}

declare i8 @llvm.sadd.sat.i8(i8, i8) #0
declare i16 @llvm.sadd.sat.i16(i16, i16) #0
declare <2 x i16> @llvm.sadd.sat.v2i16(<2 x i16>, <2 x i16>) #0
declare <3 x i16> @llvm.sadd.sat.v3i16(<3 x i16>, <3 x i16>) #0
declare <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16>, <4 x i16>) #0
declare i32 @llvm.sadd.sat.i32(i32, i32) #0
declare <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32>, <2 x i32>) #0
declare i64 @llvm.sadd.sat.i64(i64, i64) #0
