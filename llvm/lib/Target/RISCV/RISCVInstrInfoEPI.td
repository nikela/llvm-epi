//=- RISCVInstrInfoV.td - Zeou-Extension RISCV instructions -*- tblgen-*----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target Specific DAG nodes
//===----------------------------------------------------------------------===//

def riscv_vmv_x_s : SDNode<"RISCVISD::VMV_X_S",
                           SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisVec<1>,
                                                SDTCisInt<1>]>>;

def riscv_extract_vector_elt : SDNode<"RISCVISD::EXTRACT_VECTOR_ELT",
                                      SDTypeProfile<1, 2,
                                       [SDTCisInt<0>, SDTCisVec<1>,
                                       SDTCisInt<1>, SDTCisPtrTy<2>]>>;

def riscv_sign_extend_vector : SDNode<"RISCVISD::SIGN_EXTEND_VECTOR",
                                      SDTypeProfile<1, 1,
                                       [SDTCisVec<0>, SDTCisVec<1>]>>;

def riscv_zero_extend_vector : SDNode<"RISCVISD::ZERO_EXTEND_VECTOR",
                                      SDTypeProfile<1, 1,
                                       [SDTCisVec<0>, SDTCisVec<1>]>>;

def riscv_trunc_vector : SDNode<"RISCVISD::TRUNCATE_VECTOR",
                                SDTypeProfile<1, 1,
                                 [SDTCisVec<0>, SDTCisVec<1>]>>;

def riscv_sign_extend_inreg : SDNode<"RISCVISD::SIGN_EXTEND_BITS_INREG",
                                     SDTypeProfile<1, 2,
                                      [SDTCisSameAs<0, 1>,
                                      SDTCisVec<0>, SDTCisInt<0>,
                                      SDTCisInt<2>]>>;

def riscv_vlseg2 : SDNode<"RISCVISD::VLSEG2",
                          SDTypeProfile<1, 3, [SDTCisPtrTy<1>, SDTCisInt<2>,
                                               SDTCisInt<3>]>,
                          [SDNPHasChain, SDNPMayLoad]>;

def riscv_vsseg2 : SDNode<"RISCVISD::VSSEG2",
                          SDTypeProfile<0, 4, [SDTCisPtrTy<1>, SDTCisInt<2>,
                                               SDTCisInt<3>]>,
                          [SDNPHasChain, SDNPMayStore]>;

def riscv_vzip2 : SDNode<"RISCVISD::VZIP2",
                         SDTypeProfile<1, 3, [SDTCisSameAs<1, 2>,
                                             SDTCisVec<1>, SDTCisInt<3>]>>;

def riscv_vunzip2 : SDNode<"RISCVISD::VUNZIP2",
                           SDTypeProfile<1, 3, [SDTCisSameAs<1, 2>,
                                               SDTCisVec<1>, SDTCisInt<3>]>>;

def riscv_vtrn : SDNode<"RISCVISD::VTRN",
                        SDTypeProfile<1, 3, [SDTCisSameAs<1, 2>, SDTCisVec<1>,
                                            SDTCisInt<3>]>>;

// Vector reduction nodes.
def vecreduce_and : SDNode<"ISD::VECREDUCE_AND", SDTVecReduce>;
def vecreduce_or  : SDNode<"ISD::VECREDUCE_OR", SDTVecReduce>;
def vecreduce_xor : SDNode<"ISD::VECREDUCE_XOR", SDTVecReduce>;

// Vector fp reduction.
def SDTVecReduceFP : SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVec<1>,
                                          SDTCisFP<1>]>;

def vecreduce_fadd : SDNode<"ISD::VECREDUCE_FADD", SDTVecReduceFP>;
def vecreduce_fmax : SDNode<"ISD::VECREDUCE_FMAX", SDTVecReduceFP>;
def vecreduce_fmin : SDNode<"ISD::VECREDUCE_FMIN", SDTVecReduceFP>;

// Vector fp reduction with initial accumulator value (v2).
def SDTVecReduceFPV2 : SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisFP<1>,
                                            SDTCisVec<2>, SDTCisFP<2>]>;

def vecreduce_strict_fadd : SDNode<"ISD::VECREDUCE_STRICT_FADD", SDTVecReduceFPV2>;

def splat_vector : SDNode<"ISD::SPLAT_VECTOR",
                           SDTypeProfile<1, 1, [SDTCisVec<0>]>>;

//===----------------------------------------------------------------------===//
// EPI custom instructions
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs2, vs1, vm
class VCustom_VV_VV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVCustomInstVV<funct6, opv, (outs VRegOp:$vd),
                     (ins VRegOp:$vs2, VRegOp:$vs1, VMaskOp:$vm),
                     opcodestr, "$vd, $vs2, $vs1${vm}">;
}

// FIXME: Should we use some other predicate?
let Predicates = [HasStdExtV] in {

def VZIP2_VV : VCustom_VV_VV<0b110101, OPIVV, "vzip2.vv">;
def VUNZIP2_VV : VCustom_VV_VV<0b110110, OPIVV, "vunzip2.vv">;
def VTRN_VV : VCustom_VV_VV<0b110111, OPIVV, "vtrn.vv">;

}

//===----------------------------------------------------------------------===//

class VMask<bits<1> vtype>
{
  bits<1> Value = vtype;
}

def vmask_all_lanes : VMask<0b1>;
def vmask_only_true : VMask<0b0>;

class EPILookupIntrinsic<string basename> {
  Intrinsic I = !cast<Intrinsic>("int_epi_" # basename);
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Join strings in list using separator and ignoring empty elements
class Join<list<string> strings, string separator> {
  string ret = !foldl(!head(strings), !tail(strings), a, b,
                      !cond(
                        !and(!empty(a), !empty(b)) : "",
                        !empty(a) : b,
                        !empty(b) : a,
                        1 : a#separator#b));
}

//===----------------------------------------------------------------------===//
// Common definitions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    Uses = [VL] in
def PseudoReadVL : Pseudo<(outs GPR:$rd),
                          (ins), [], "rdvl", "$rd">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1,
     usesCustomInserter = 1 in
def PseudoVSCALE : Pseudo<(outs GPR:$rd), (ins), [], "">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    Uses = [VTYPE] in
def PseudoReadVTYPE : Pseudo<(outs GPR:$rd),
                             (ins), [], "rdvtype", "$rd">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0 in
def PseudoReadVLENB : Pseudo<(outs GPR:$rd),
                             (ins), [], "rdvlenb", "$rd">;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, Defs = [VL, VTYPE] in {
def PseudoVSETVLI : Pseudo<(outs GPR:$rd), (ins GPR:$rs1, VTypeIOp:$vtypei), []>;
def PseudoVSETVL : Pseudo<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2), []>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, usesCustomInserter = 1,
    Uses = [VL, VTYPE] in
foreach vlmul = [1, 2, 4, 8] in {
def "PseudoVMCLR_M"#vlmul : Pseudo<(outs VR:$rd), (ins GPR:$vl, ixlenimm:$sew), []>;
def "PseudoVMSET_M"#vlmul : Pseudo<(outs VR:$rd), (ins GPR:$vl, ixlenimm:$sew), []>;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions we need for SPILL and RELOAD
//===----------------------------------------------------------------------===//

multiclass pseudo_spill_reload<VReg reg_class, int vlmul> {

let hasSideEffects = 1, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1,
    Uses = [VL, VTYPE] in
def PseudoVSPILL_M#vlmul : Pseudo<(outs), (ins reg_class:$rs1, GPR:$rs2), []>;

let hasSideEffects = 1, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1,
    Uses = [VL, VTYPE] in
def PseudoVRELOAD_M#vlmul : Pseudo<(outs reg_class:$rs1), (ins GPR:$rs2), []>;

}

defm "" : pseudo_spill_reload<VR, 1>;
defm "" : pseudo_spill_reload<VRM2, 2>;
defm "" : pseudo_spill_reload<VRM4, 4>;
defm "" : pseudo_spill_reload<VRM8, 8>;

// Tuples.
let hasSideEffects = 1, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1,
    Uses = [VL, VTYPE] in
def PseudoVSPILL_2xM1 : Pseudo<(outs), (ins VRTuple:$rs1, GPR:$rs2), []>;

let hasSideEffects = 1, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1,
    Uses = [VL, VTYPE] in
def PseudoVRELOAD_2xM1 : Pseudo<(outs VRTuple:$rs1), (ins GPR:$rs2), []>;

}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

class VectorTypeInfo<ValueType Vec, ValueType Mas, int Sew, VReg Reg>
{
  ValueType Vector = Vec;
  ValueType Mask = Mas;
  int SEW = Sew;
  VReg RegClass = Reg;
}

class GroupVectorTypeInfo<ValueType Vec, ValueType VecM1, ValueType Mas,
                          int Sew, VReg Reg>
    : VectorTypeInfo<Vec, Mas, Sew, Reg>
{
  ValueType VectorM1 = VecM1;
}

class FloatVectorTypeInfo<ValueType Vec, ValueType Mas, int Sew, VReg Reg,
                          ValueType Scal, RegisterClass ScalarReg>
    : VectorTypeInfo<Vec, Mas, Sew, Reg>
{
  ValueType Scalar = Scal;
  RegisterClass ScalarRegClass = ScalarReg;
}

class GroupFloatVectorTypeInfo<ValueType Vec, ValueType VecM1, ValueType Mas,
                               int Sew, VReg Reg, ValueType Scal,
                               RegisterClass ScalarReg>
    : FloatVectorTypeInfo<Vec, Mas, Sew, Reg, Scal, ScalarReg>
{
  ValueType VectorM1 = VecM1;
}

class VectorTypeInfoToWide<VectorTypeInfo vti, VectorTypeInfo wti>
{
  VectorTypeInfo Vti = vti;
  VectorTypeInfo Wti = wti;
}

class FloatVectorTypeInfoToWide<FloatVectorTypeInfo fvti, FloatVectorTypeInfo fwti>
{
  FloatVectorTypeInfo FVti = fvti;
  FloatVectorTypeInfo FWti = fwti;
}

defset list<VectorTypeInfo> AllVectors = {

  defset list<VectorTypeInfo> AllIntegerVectors = {
    defset list<VectorTypeInfo> NoGroupIntegerVectors = {
//    def Vtype1xi8  : VectorTypeInfo<nxv1i8,  nxv1i1, 8,  VR>; // FIXME illegal type
//    def Vtype2xi8  : VectorTypeInfo<nxv2i8,  nxv2i1, 8,  VR>; // FIXME illegal type
//    def Vtype4xi8  : VectorTypeInfo<nxv4i8,  nxv4i1, 8,  VR>; // FIXME illegal type
      def Vtype8xi8  : VectorTypeInfo<nxv8i8,  nxv8i1, 8,  VR>;

//    def Vtype1xi16 : VectorTypeInfo<nxv1i16, nxv1i1, 16, VR>; // FIXME illegal type
//    def Vtype2xi16 : VectorTypeInfo<nxv2i16, nxv2i1, 16, VR>; // FIXME illegal type
      def Vtype4xi16 : VectorTypeInfo<nxv4i16, nxv4i1, 16, VR>;

//    def Vtype1xi32 : VectorTypeInfo<nxv1i32, nxv1i1, 32, VR>; // FIXME illegal type
      def Vtype2xi32 : VectorTypeInfo<nxv2i32, nxv2i1, 32, VR>;

      def Vtype1xi64 : VectorTypeInfo<nxv1i64, nxv1i1, 64, VR>;
    }

    defset list<GroupVectorTypeInfo> GroupIntegerVectors = {
      def Vtype16xi8  : GroupVectorTypeInfo<nxv16i8,  nxv8i8,  nxv16i1, 8,  VRM2>;
      def Vtype32xi8  : GroupVectorTypeInfo<nxv32i8,  nxv8i8,  nxv32i1, 8,  VRM4>;
//    def Vtype64xi8  : GroupVectorTypeInfo<nxv64i8,  nxv8i8,  nxv64i1, 8,  VRM8>; // FIXME undefined type

      def Vtype8xi16  : GroupVectorTypeInfo<nxv8i16,  nxv4i16, nxv8i1,  16, VRM2>;
      def Vtype16xi16 : GroupVectorTypeInfo<nxv16i16, nxv4i16, nxv16i1, 16, VRM4>;
      def Vtype32xi16 : GroupVectorTypeInfo<nxv32i16, nxv4i16, nxv32i1, 16, VRM8>;

      def Vtype4xi32  : GroupVectorTypeInfo<nxv4i32,  nxv2i32, nxv4i1,  32, VRM2>;
      def Vtype8xi32  : GroupVectorTypeInfo<nxv8i32,  nxv2i32, nxv8i1,  32, VRM4>;
      def Vtype16xi32 : GroupVectorTypeInfo<nxv16i32, nxv2i32, nxv16i1, 32, VRM8>;

      def Vtype2xi64  : GroupVectorTypeInfo<nxv2i64,  nxv1i64, nxv2i1,  64, VRM2>;
      def Vtype4xi64  : GroupVectorTypeInfo<nxv4i64,  nxv1i64, nxv4i1,  64, VRM4>;
      def Vtype8xi64  : GroupVectorTypeInfo<nxv8i64,  nxv1i64, nxv8i1,  64, VRM8>;
    }
  }

  defset list<FloatVectorTypeInfo> AllFloatVectors = {
    defset list<FloatVectorTypeInfo> NoGroupFloatVectors = {
//    def Vtype1xf32 : FloatVectorTypeInfo<nxv1f32, nxv1i1, 32, VR, f32, FPR32>; // FIXME illegal type
      def Vtype2xf32 : FloatVectorTypeInfo<nxv2f32, nxv2i1, 32, VR, f32, FPR32>;

      def Vtype1xf64 : FloatVectorTypeInfo<nxv1f64, nxv1i1, 64, VR, f64, FPR64>;
    }

    defset list<GroupFloatVectorTypeInfo> GroupFloatVectors = {
      def Vtype4xf32  : GroupFloatVectorTypeInfo<nxv4f32,  nxv2f32, nxv4i1,  32, VRM2, f32, FPR32>;
      def Vtype8xf32  : GroupFloatVectorTypeInfo<nxv8f32,  nxv2f32, nxv8i1,  32, VRM4, f32, FPR32>;
      def Vtype16xf32 : GroupFloatVectorTypeInfo<nxv16f32, nxv2f32, nxv16i1, 32, VRM8, f32, FPR32>;

      def Vtype2xf64  : GroupFloatVectorTypeInfo<nxv2f64,  nxv1f64, nxv2i1,  64, VRM2, f64, FPR64>;
      def Vtype4xf64  : GroupFloatVectorTypeInfo<nxv4f64,  nxv1f64, nxv4i1,  64, VRM4, f64, FPR64>;
      def Vtype8xf64  : GroupFloatVectorTypeInfo<nxv8f64,  nxv1f64, nxv8i1,  64, VRM8, f64, FPR64>;
    }
  }
}

defvar NoGroupVectors = !listconcat(NoGroupIntegerVectors, NoGroupFloatVectors);

// This functor is used to obtain the int vector type that has the same SEW and
// multiplier as the input parameter type
class GetIntVectorTypeInfo<VectorTypeInfo vti>
{
  // Equivalent integer vector type. Eg.
  //   Vtype8xi8 → Vtype8xi8 (identity)
  //   Vtype4xf64 → Vtype4xi64
  VectorTypeInfo Vti =
    !cast<VectorTypeInfo>(
      !subst("f", "i", !cast<string>(vti))
    );
}

// This functor is used to obtain the float vector type that has the same SEW
// and multiplier as the input parameter type
class GetFloatVectorTypeInfo<VectorTypeInfo vti>
{
  // Equivalent float vector type. Eg.
  //   Vtype32xi16 → Vtype32xf16
  //   Vtype2xf32 → Vtype2xf32 (identity)
  VectorTypeInfo FVti =
    !cast<VectorTypeInfo>(
      !subst("i", "f", !cast<string>(vti))
    );
}

defset list<VectorTypeInfoToWide> AllWideableIntVectors = {
//def : VectorTypeInfoToWide<Vtype1xi8,   Vtype1xi16>; // FIXME illegal type
//def : VectorTypeInfoToWide<Vtype2xi8,   Vtype2xi16>; // FIXME illegal type
//def : VectorTypeInfoToWide<Vtype4xi8,   Vtype4xi16>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype8xi8,   Vtype8xi16>;
  def : VectorTypeInfoToWide<Vtype16xi8,  Vtype16xi16>;
  def : VectorTypeInfoToWide<Vtype32xi8,  Vtype32xi16>;

//def : VectorTypeInfoToWide<Vtype1xi16,  Vtype1xi32>; // FIXME illegal type
//def : VectorTypeInfoToWide<Vtype2xi16,  Vtype2xi32>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype4xi16,  Vtype4xi32>;
  def : VectorTypeInfoToWide<Vtype8xi16,  Vtype8xi32>;
  def : VectorTypeInfoToWide<Vtype16xi16, Vtype16xi32>;

//def : VectorTypeInfoToWide<Vtype1xi32,  Vtype1xi64>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype2xi32,  Vtype2xi64>;
  def : VectorTypeInfoToWide<Vtype4xi32,  Vtype4xi64>;
  def : VectorTypeInfoToWide<Vtype8xi32,  Vtype8xi64>;
// FIXME what about these?
//def : VectorTypeInfoToWide<Vtype1xi64,  /* FIXME Vtype1xi128 */ i1>;
//def : VectorTypeInfoToWide<Vtype2xi64,  /* FIXME Vtype2xi128 */ i1>;
//def : VectorTypeInfoToWide<Vtype4xi64,  /* FIXME Vtype4xi128 */ i1>;
}

defset list<VectorTypeInfoToWide> AllWideableIntToFloatVectors = {
//def : VectorTypeInfoToWide<Vtype1xi16,  Vtype1xi32>; // FIXME illegal type
//def : VectorTypeInfoToWide<Vtype2xi16,  Vtype2xi32>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype4xi16,  Vtype4xi32>;
  def : VectorTypeInfoToWide<Vtype8xi16,  Vtype8xi32>;
  def : VectorTypeInfoToWide<Vtype16xi16, Vtype16xi32>;

//def : VectorTypeInfoToWide<Vtype1xi32,  Vtype1xi64>; // FIXME illegal type
  def : VectorTypeInfoToWide<Vtype2xi32,  Vtype2xi64>;
  def : VectorTypeInfoToWide<Vtype4xi32,  Vtype4xi64>;
  def : VectorTypeInfoToWide<Vtype8xi32,  Vtype8xi64>;
}

defset list<FloatVectorTypeInfoToWide> AllWideableFloatVectors = {
//def : FloatVectorTypeInfoToWide<Vtype1xf32, Vtype1xf64>; // FIXME illegal type
  def : FloatVectorTypeInfoToWide<Vtype2xf32, Vtype2xf64>;
  def : FloatVectorTypeInfoToWide<Vtype4xf32, Vtype4xf64>;
  def : FloatVectorTypeInfoToWide<Vtype8xf32, Vtype8xf64>;
// FIXME what about these?
//def : FloatVectorTypeInfoToWide<Vtype1xf64, /* FIXME: Vtype1xf128 */ i1>;
//def : FloatVectorTypeInfoToWide<Vtype2xf64, /* FIXME: Vtype2xf128 */ i1>;
//def : FloatVectorTypeInfoToWide<Vtype4xf64, /* FIXME: Vtype4xf128 */ i1>;
}

class MaskTypeInfo<ValueType Mas, int Sew, int Vlmul> {
  ValueType Mask = Mas;
  // {SEW, VLMul} values set a valid VType to deal with this mask type.
  int SEW = Sew; // FIXME: computed as ELEN/vscale, to be used for loading/storing partial masks
  int VLMul = Vlmul; // Convention: Minimum VLMul where this mask type appears.
}

defset list<MaskTypeInfo> AllMasks = {
  def : MaskTypeInfo<nxv1i1,  64, 1>; // FIXME SEW should be ELEN/1
  def : MaskTypeInfo<nxv2i1,  32, 1>; // FIXME SEW should be ELEN/2
  def : MaskTypeInfo<nxv4i1,  16, 1>; // FIXME SEW should be ELEN/4
  def : MaskTypeInfo<nxv8i1,  8,  1>;  // FIXME SEW should be ELEN/8
  def : MaskTypeInfo<nxv16i1, 8,  2>;  // FIXME SEW should be ELEN/16, SEW < 8
  def : MaskTypeInfo<nxv32i1, 8,  4>;  // FIXME SEW should be ELEN/32, SEW < 8
//def : MaskTypeInfo<nxv64i1, 8,  8>;  // FIXME SEW should be ELEN/64, SEW < 8, undefined type
}

class IllegalVectorTypeInfo<ValueType Vec, VectorTypeInfo LVti, ValueType Mas,
                            int Sew, VReg Reg>
    : VectorTypeInfo<Vec, Mas, Sew, Reg>
{
  VectorTypeInfo LegalVti = LVti;
}

defset list<IllegalVectorTypeInfo> AllIllegalIntegerVectors = {
  def Vtype1xi8  : IllegalVectorTypeInfo<nxv1i8,  Vtype1xi64, nxv1i1, 8,  VR>;
  def Vtype2xi8  : IllegalVectorTypeInfo<nxv2i8,  Vtype2xi32, nxv2i1, 8,  VR>;
  def Vtype4xi8  : IllegalVectorTypeInfo<nxv4i8,  Vtype4xi16, nxv4i1, 8,  VR>;

  def Vtype1xi16 : IllegalVectorTypeInfo<nxv1i16, Vtype1xi64, nxv1i1, 16, VR>;
  def Vtype2xi16 : IllegalVectorTypeInfo<nxv2i16, Vtype2xi32, nxv2i1, 16, VR>;

  def Vtype1xi32 : IllegalVectorTypeInfo<nxv1i32, Vtype1xi64, nxv1i1, 32, VR>;
}

class VRegToWide<VReg Reg, VReg WideReg> {
  VReg RegClass = Reg;
  VReg WideRegClass = WideReg;
}

defset list<VRegToWide> AllVRegToWidePairs = {
  def : VRegToWide<VR, VRM2>;
  def : VRegToWide<VRM2, VRM4>;
  def : VRegToWide<VRM4, VRM8>;
}

defvar AllEEW = [8, 16, 32, 64];

def EPIIntrClassID : GenericEnum {
  let FilterClass = "EPIIntrinsicClassID";
}

def EPIIntrinsicsTable : GenericTable {
  let FilterClass = "EPIIntrinsic";
  let CppTypeName = "EPIIntrinsicInfo";
  let Fields = [ "IntrinsicID", "ClassID", "ExtendOperand", "MaskOperand",
                 "GVLOperand" ];
  let PrimaryKey = [ "IntrinsicID" ];
  let PrimaryKeyName = "getEPIIntrinsicInfo";

  GenericEnum TypeOf_ClassID = EPIIntrClassID;
}

class EPIPseudo {
  Pseudo Pseudo = !cast<Pseudo>(NAME);
  Instruction BaseInstr;
  bits<8> VLIndex;
  bits<8> SEWIndex;
  bits<8> MergeOpIndex;
  bits<8> VLMul;
}

def EPIPseudosTable : GenericTable {
  let FilterClass = "EPIPseudo";
  let CppTypeName = "EPIPseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr", "VLIndex", "SEWIndex", "MergeOpIndex", "VLMul" ];
  let PrimaryKey = [ "Pseudo" ];
  let PrimaryKeyName = "getEPIPseudoInfo";
}


multiclass pseudo_nullary<VReg result_reg_class,
                          int vlmul,
                          string constraints> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins result_reg_class:$merge,
                                 VMaskOp:$vm,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_nullary_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _V : pseudo_nullary<evr, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_unary<VReg result_reg_class,
                        VReg op_reg_class,
                        int vlmul,
                        string constraints> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins result_reg_class:$merge,
                                 op_reg_class:$rs2, VMaskOp:$vm,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

// Special case for masking that does not have a merge operand.
multiclass pseudo_unary_nomerge<VReg result_reg_class,
                                VReg op_reg_class,
                                int vlmul,
                                string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op_reg_class:$rs2, VMaskOp:$vm,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

// Special case for masking that does not have a merge operand nor a mask.
multiclass pseudo_unary_nomask<VReg result_reg_class,
                               DAGOperand op_kind,
                               int vlmul,
                               string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 2, SEWIndex = 3, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op_kind:$rs2,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_unary_v_m {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.VLMul in
    {
      defm _M : pseudo_unary<evr, VR, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_unary_v_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _V : pseudo_unary<evr, evr, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_unary_v_v_x_i_nomask
{
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _V : pseudo_unary_nomask<evr, evr, evr.VLMul>;
      defm _X : pseudo_unary_nomask<evr, GPR, evr.VLMul>;
      defm _I : pseudo_unary_nomask<evr, simm5, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary_v_f_nomask {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    let VLMul = evr.VLMul in
    {
      defm _F : pseudo_unary_nomask<evr, FPR64, evr.VLMul>;
    }
  }
}

multiclass pseudo_unary_w_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.RegClass.VLMul in
    {
      defm _V : pseudo_unary<evr.WideRegClass, evr.RegClass,
                             evr.RegClass.VLMul, constraints>;
    }
  }
}

multiclass pseudo_unary_v_w {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.RegClass.VLMul in
    {
      defm _W : pseudo_unary<evr.RegClass, evr.WideRegClass,
                             evr.RegClass.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary<VReg result_reg_class,
                         VReg op1_reg_class,
                         DAGOperand op2_kind,
                         int vlmul,
                         string constraints> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins result_reg_class:$merge,
                                 op1_reg_class:$rs2, op2_kind:$rs1,
                                 VMaskOp:$vm, GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

// Special case for masking that does not have a merge operand nor a mask.
multiclass pseudo_binary_nomask<VReg result_reg_class,
                                VReg op1_reg_class,
                                DAGOperand op2_kind,
                                int vlmul, string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op1_reg_class:$rs2, op2_kind:$rs1,
                                 GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_binary_mask_in<VReg result_reg_class,
                                 VReg op1_reg_class,
                                 DAGOperand op2_kind,
                                 int vlmul,
                                 string constraints> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                            (ins op1_reg_class:$rs2, op2_kind:$rs1,
                                 VMV0:$maskop, GPR:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_binary_v_vv {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vx<bit force_earlyclobber = 0> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!or(force_earlyclobber, !gt(evr.VLMul, 1)),
                             "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vs {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VS : pseudo_binary<evr, evr, evr, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul, constraints>;
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vx_vi<DAGOperand imm_kind = simm5,
                                 bit force_earlyclobber = 0> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!or(force_earlyclobber, !gt(evr.VLMul, 1)),
                             "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul, constraints>;
      defm _VI : pseudo_binary<evr, evr, imm_kind, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vv_vx_vi<DAGOperand imm_kind = simm5,
                                    bit force_earlyclobber = 0> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!or(force_earlyclobber, !gt(evr.VLMul, 1)),
                             "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul, constraints>;
      defm _VX : pseudo_binary<evr, evr, GPR, evr.VLMul, constraints>;
      defm _VI : pseudo_binary<evr, evr, imm_kind, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_wv_wx_wi<DAGOperand imm_kind = simm5> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.RegClass.VLMul in
    {
      defm _WV : pseudo_binary<evr.RegClass, evr.WideRegClass, evr.RegClass,
                               evr.RegClass.VLMul, constraints>;
      defm _WX : pseudo_binary<evr.RegClass, evr.WideRegClass, GPR,
                               evr.RegClass.VLMul, constraints>;
      defm _WI : pseudo_binary<evr.RegClass, evr.WideRegClass, imm_kind,
                               evr.RegClass.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vvm_vxm {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VVM : pseudo_binary_mask_in<evr, evr, evr, evr.VLMul, constraints>;
      defm _VXM : pseudo_binary_mask_in<evr, evr, GPR, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vvm_vxm_vim {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VVM : pseudo_binary_mask_in<evr, evr, evr, evr.VLMul, constraints>;
      defm _VXM : pseudo_binary_mask_in<evr, evr, GPR, evr.VLMul, constraints>;
      defm _VIM : pseudo_binary_mask_in<evr, evr, simm5, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vvm_vxm {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.VLMul in
    {
      // These instructions generate a mask register so their result is a VR.
      defm _VVM : pseudo_binary_mask_in<VR, evr, evr, evr.VLMul, constraints>;
      defm _VXM : pseudo_binary_mask_in<VR, evr, GPR, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vvm_vxm_vim {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.VLMul in
    {
      // These instructions generate a mask register so their result is a VR.
      defm _VVM : pseudo_binary_mask_in<VR, evr, evr, evr.VLMul, constraints>;
      defm _VXM : pseudo_binary_mask_in<VR, evr, GPR, evr.VLMul, constraints>;
      defm _VIM : pseudo_binary_mask_in<VR, evr, simm5, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_w_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.RegClass.VLMul in
    {
      defm _VV : pseudo_binary<evr.WideRegClass, evr.RegClass, evr.RegClass,
                               evr.RegClass.VLMul, constraints>;
      defm _VX : pseudo_binary<evr.WideRegClass, evr.RegClass, GPR,
                               evr.RegClass.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_w_wv_wx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.RegClass.VLMul in
    {
      defm _WV : pseudo_binary<evr.WideRegClass, evr.WideRegClass, evr.RegClass,
                               evr.RegClass.VLMul, constraints>;
      defm _WX : pseudo_binary<evr.WideRegClass, evr.WideRegClass, GPR,
                               evr.RegClass.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<VR, evr, evr, evr.VLMul, constraints>;
      defm _VX : pseudo_binary<VR, evr, GPR, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vv_vx_nomask {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary_nomask<VR, evr, evr, evr.VLMul, constraints>;
      defm _VX : pseudo_binary_nomask<VR, evr, GPR, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VX : pseudo_binary<VR, evr, GPR, evr.VLMul, constraints>;
      defm _VI : pseudo_binary<VR, evr, simm5, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vv_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<VR, evr, evr, evr.VLMul, constraints>;
      defm _VX : pseudo_binary<VR, evr, GPR, evr.VLMul, constraints>;
      defm _VI : pseudo_binary<VR, evr, simm5, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vv_vx_vi_nomask {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary_nomask<VR, evr, evr, evr.VLMul, constraints>;
      defm _VX : pseudo_binary_nomask<VR, evr, GPR, evr.VLMul, constraints>;
      defm _VI : pseudo_binary_nomask<VR, evr, simm5, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VF : pseudo_binary<evr, evr, FPR64, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vfm {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VFM : pseudo_binary_mask_in<evr, evr, FPR64, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<evr, evr, evr, evr.VLMul, constraints>;
      defm _VF : pseudo_binary<evr, evr, FPR64, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_w_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.RegClass.VLMul in
    {
      defm _VV : pseudo_binary<evr.WideRegClass, evr.RegClass, evr.RegClass,
                               evr.RegClass.VLMul, constraints>;
      defm _VF : pseudo_binary<evr.WideRegClass, evr.RegClass, FPR64,
                               evr.RegClass.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_w_wv_wf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegToWidePairs in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = evr.RegClass.VLMul in
    {
      defm _WV : pseudo_binary<evr.WideRegClass, evr.WideRegClass, evr.RegClass,
                               evr.RegClass.VLMul, constraints>;
      defm _WF : pseudo_binary<evr.WideRegClass, evr.WideRegClass, FPR64,
                               evr.RegClass.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VF : pseudo_binary<VR, evr, FPR64, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_binary<VR, evr, evr, evr.VLMul, constraints>;
      defm _VF : pseudo_binary<VR, evr, FPR64, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_ternary<VReg result_reg_class,
                          VReg op1_reg_class,
                          RegisterClass op2_reg_class,
                          int vlmul,
                          string constraints> {
  let Constraints = Join<["$rd = $rs3", constraints], ",">.ret,
      Uses = [VL, VTYPE],
      VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
      BaseInstr = !cast<Instruction>(!subst("Pseudo", "", NAME)) in
    def "_M"#vlmul : Pseudo<(outs result_reg_class:$rd),
                                 (ins result_reg_class:$rs3, op1_reg_class:$rs1,
                                      op2_reg_class:$rs2, VMaskOp:$vm,
                                      GPR:$vl, ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
}

multiclass pseudo_ternary_v_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_ternary<evr, evr, evr, evr.VLMul, constraints>;
      defm _VX : pseudo_ternary<evr, GPR, evr, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_ternary_v_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach evr = AllVRegs in
  {
    defvar constraints = !if(!gt(evr.VLMul, 1), "@earlyclobber $rd", "");
    let VLMul = evr.VLMul in
    {
      defm _VV : pseudo_ternary<evr, evr, evr, evr.VLMul, constraints>;
      defm _VF : pseudo_ternary<evr, FPR64, evr, evr.VLMul, constraints>;
    }
  }
}

multiclass pseudo_unary_x_m {
  // A different pseudo is defined for each VLMul value, but a VR operand
  // is used in all of them.
  foreach Vlmul = [1, 2, 4, 8] in
  {
    let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
        VLMul = Vlmul in
    {
      defm _M : pseudo_unary_nomerge<GPR, VR, Vlmul>;
    }
  }
}

multiclass pseudo_binary_m_mm {
  // A different pseudo is defined for each VLMul value, but VR operands
  // are used in all of them.
  foreach Vlmul = [1, 2, 4, 8] in
  {
    let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
        VLMul = Vlmul in
    {
      defm _MM : pseudo_binary_nomask<VR, VR, VR, Vlmul>;
    }
  }
}

let Predicates = [HasStdExtV] in {

defm PseudoVADD        : pseudo_binary_v_vv_vx_vi;
defm PseudoVSUB        : pseudo_binary_v_vv_vx;
defm PseudoVRSUB       : pseudo_binary_v_vx_vi;
defm PseudoVMINU       : pseudo_binary_v_vv_vx;
defm PseudoVMIN        : pseudo_binary_v_vv_vx;
defm PseudoVMAXU       : pseudo_binary_v_vv_vx;
defm PseudoVMAX        : pseudo_binary_v_vv_vx;
defm PseudoVAND        : pseudo_binary_v_vv_vx_vi;
defm PseudoVOR         : pseudo_binary_v_vv_vx_vi;
defm PseudoVXOR        : pseudo_binary_v_vv_vx_vi;

defm PseudoVRGATHER    : pseudo_binary_v_vv_vx_vi<uimm5, /* force_earlyclobber */ 1>;
defm PseudoVSLIDEUP    : pseudo_binary_v_vx_vi<uimm5, /* force_earlyclobber */ 1>;
defm PseudoVSLIDEDOWN  : pseudo_binary_v_vx_vi<uimm5>;

defm PseudoVMSEQ       : pseudo_binary_m_vv_vx_vi;
defm PseudoVMSNE       : pseudo_binary_m_vv_vx_vi;
defm PseudoVMSLTU      : pseudo_binary_m_vv_vx;
defm PseudoVMSLT       : pseudo_binary_m_vv_vx;
defm PseudoVMSLEU      : pseudo_binary_m_vv_vx_vi;
defm PseudoVMSLE       : pseudo_binary_m_vv_vx_vi;

defm PseudoVMSGTU      : pseudo_binary_m_vx_vi;
defm PseudoVMSGT       : pseudo_binary_m_vx_vi;

defm PseudoVSADDU      : pseudo_binary_v_vv_vx_vi;
defm PseudoVSADD       : pseudo_binary_v_vv_vx_vi;
defm PseudoVSSUBU      : pseudo_binary_v_vv_vx;
defm PseudoVSSUB       : pseudo_binary_v_vv_vx;
defm PseudoVAADD       : pseudo_binary_v_vv_vx;
// FIXME missing PseudoVAADDU
defm PseudoVSLL        : pseudo_binary_v_vv_vx_vi<uimm5>;
defm PseudoVASUB       : pseudo_binary_v_vv_vx;
defm PseudoVSMUL       : pseudo_binary_v_vv_vx;
defm PseudoVSRL        : pseudo_binary_v_vv_vx_vi<uimm5>;
defm PseudoVSRA        : pseudo_binary_v_vv_vx_vi<uimm5>;
defm PseudoVSSRL       : pseudo_binary_v_vv_vx_vi<uimm5>;
defm PseudoVSSRA       : pseudo_binary_v_vv_vx_vi<uimm5>;

defm PseudoVNSRL       : pseudo_binary_v_wv_wx_wi<uimm5>;
defm PseudoVNSRA       : pseudo_binary_v_wv_wx_wi<uimm5>;

//FIXME missing PseudoVNCLIP (uimm5)
//FIXME missing PseudoVNCLIPU (uimm5)

defm PseudoVMV_V       : pseudo_unary_v_v_x_i_nomask;
defm PseudoVMERGE      : pseudo_binary_v_vvm_vxm_vim;

// FIXME EDIV instructions disabled
//defm PseudoVDOTU       : pseudo_binary_v_vv;
//defm PseudoVDOT        : pseudo_binary_v_vv;

defm PseudoVREDSUM     : pseudo_binary_v_vs;
defm PseudoVREDAND     : pseudo_binary_v_vs;
defm PseudoVREDOR      : pseudo_binary_v_vs;
defm PseudoVREDXOR     : pseudo_binary_v_vs;
defm PseudoVREDMINU    : pseudo_binary_v_vs;
defm PseudoVREDMIN     : pseudo_binary_v_vs;
defm PseudoVREDMAXU    : pseudo_binary_v_vs;
defm PseudoVREDMAX     : pseudo_binary_v_vs;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
    Uses = [VL, VTYPE], VLMul = VR.VLMul in
{
  let VLIndex = -1, SEWIndex = 2, MergeOpIndex = -1, BaseInstr = VMV_X_S in
    def PseudoVMV_X_S : Pseudo<(outs GPR:$rd),
                               (ins VR:$rs2, ixlenimm:$sew),
                               []>,
                        EPIPseudo;
  let Constraints = "$rd = $merge", VLIndex = 3, SEWIndex = 4, MergeOpIndex = 1,
      BaseInstr = VMV_S_X in
    def PseudoVMV_S_X : Pseudo<(outs VR:$rd),
                               (ins VR:$merge, GPR:$rs1, GPR:$vl,
                                    ixlenimm:$sew),
                               []>,
                        EPIPseudo;
}

defm PseudoVSLIDE1UP   : pseudo_binary_v_vx</* force_earlyclobber */ 1>;
defm PseudoVSLIDE1DOWN : pseudo_binary_v_vx;

defm PseudoVDIVU       : pseudo_binary_v_vv_vx;
defm PseudoVDIV        : pseudo_binary_v_vv_vx;
defm PseudoVREMU       : pseudo_binary_v_vv_vx;
defm PseudoVREM        : pseudo_binary_v_vv_vx;
defm PseudoVMULHU      : pseudo_binary_v_vv_vx;
defm PseudoVMUL        : pseudo_binary_v_vv_vx;
defm PseudoVMULHSU     : pseudo_binary_v_vv_vx;
defm PseudoVMULH       : pseudo_binary_v_vv_vx;

defm PseudoVMADD       : pseudo_ternary_v_vv_vx;
defm PseudoVNMSUB      : pseudo_ternary_v_vv_vx;
defm PseudoVMACC       : pseudo_ternary_v_vv_vx;
defm PseudoVNMSAC      : pseudo_ternary_v_vv_vx;

defm PseudoVWADDU      : pseudo_binary_w_vv_vx;
defm PseudoVWADDU      : pseudo_binary_w_wv_wx;
defm PseudoVWADD       : pseudo_binary_w_vv_vx;
defm PseudoVWADD       : pseudo_binary_w_wv_wx;
defm PseudoVWSUBU      : pseudo_binary_w_vv_vx;
defm PseudoVWSUBU      : pseudo_binary_w_wv_wx;
defm PseudoVWSUB       : pseudo_binary_w_vv_vx;
defm PseudoVWSUB       : pseudo_binary_w_wv_wx;

defm PseudoVADC        : pseudo_binary_v_vvm_vxm_vim;
defm PseudoVMADC       : pseudo_binary_m_vvm_vxm_vim;
defm PseudoVMADC       : pseudo_binary_m_vv_vx_vi_nomask;

defm PseudoVSBC        : pseudo_binary_v_vvm_vxm;
defm PseudoVMSBC       : pseudo_binary_m_vvm_vxm;
defm PseudoVMSBC       : pseudo_binary_m_vv_vx_nomask;

defm PseudoVWMULU      : pseudo_binary_w_vv_vx;
defm PseudoVWMULSU     : pseudo_binary_w_vv_vx;
defm PseudoVWMUL       : pseudo_binary_w_vv_vx;

defm PseudoVFADD       : pseudo_binary_v_vv_vf;
defm PseudoVFREDSUM    : pseudo_binary_v_vs;
defm PseudoVFSUB       : pseudo_binary_v_vv_vf;
defm PseudoVFREDOSUM   : pseudo_binary_v_vs;
defm PseudoVFMIN       : pseudo_binary_v_vv_vf;
defm PseudoVFREDMIN    : pseudo_binary_v_vs;
defm PseudoVFMAX       : pseudo_binary_v_vv_vf;
defm PseudoVFREDMAX    : pseudo_binary_v_vs;
defm PseudoVFSGNJ      : pseudo_binary_v_vv_vf;
defm PseudoVFSGNJN     : pseudo_binary_v_vv_vf;
defm PseudoVFSGNJX     : pseudo_binary_v_vv_vf;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1,
    Uses = [VL, VTYPE], VLMul = VR.VLMul, MergeOpIndex = -1 in
{
  let VLIndex = -1, SEWIndex = 2, BaseInstr = VFMV_F_S in
    def PseudoVFMV_F_S : Pseudo<(outs FPR64:$rd),
                                (ins VR:$rs2, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
  let Constraints = "$rd = $merge", VLIndex = 3, SEWIndex = 4, MergeOpIndex = 1,
      BaseInstr = VFMV_S_F in
    def PseudoVFMV_S_F : Pseudo<(outs VR:$rd),
                                (ins VR:$merge, FPR64:$rs2, GPR:$vl,
                                     ixlenimm:$sew),
                                []>,
                         EPIPseudo;
}

defm PseudoVMFEQ       : pseudo_binary_m_vv_vf;
defm PseudoVMFLE       : pseudo_binary_m_vv_vf;
defm PseudoVMFLT       : pseudo_binary_m_vv_vf;
defm PseudoVMFNE       : pseudo_binary_m_vv_vf;

defm PseudoVMFGT       : pseudo_binary_m_vf;
defm PseudoVMFGE       : pseudo_binary_m_vf;

defm PseudoVFDIV       : pseudo_binary_v_vv_vf;
defm PseudoVFRDIV      : pseudo_binary_v_vf;
defm PseudoVFMUL       : pseudo_binary_v_vv_vf;

defm PseudoVFMV_V      : pseudo_unary_v_f_nomask;
defm PseudoVFMERGE     : pseudo_binary_v_vfm;

defm PseudoVFMADD      : pseudo_ternary_v_vv_vf;
defm PseudoVFMSUB      : pseudo_ternary_v_vv_vf;
defm PseudoVFMACC      : pseudo_ternary_v_vv_vf;
defm PseudoVFMSAC      : pseudo_ternary_v_vv_vf;

defm PseudoVFNMADD     : pseudo_ternary_v_vv_vf;
defm PseudoVFNMSUB     : pseudo_ternary_v_vv_vf;
defm PseudoVFNMACC     : pseudo_ternary_v_vv_vf;
defm PseudoVFNMSAC     : pseudo_ternary_v_vv_vf;

defm PseudoVFWADD      : pseudo_binary_w_vv_vf;
defm PseudoVFWADD      : pseudo_binary_w_wv_wf;
defm PseudoVFWSUB      : pseudo_binary_w_vv_vf;
defm PseudoVFWSUB      : pseudo_binary_w_wv_wf;
defm PseudoVFWMUL      : pseudo_binary_w_vv_vf;
// FIXME EDIV instructions disabled
//defm PseudoVFDOT       : pseudo_binary_v_vv;

defm PseudoVFSQRT      : pseudo_unary_v_v;

defm PseudoVIOTA       : pseudo_unary_v_m;

defm PseudoVFCVT_XU_F  : pseudo_unary_v_v;
defm PseudoVFCVT_X_F   : pseudo_unary_v_v;
defm PseudoVFCVT_F_XU  : pseudo_unary_v_v;
defm PseudoVFCVT_F_X   : pseudo_unary_v_v;

defm PseudoVFWCVT_XU_F : pseudo_unary_w_v;
defm PseudoVFWCVT_X_F  : pseudo_unary_w_v;
defm PseudoVFWCVT_F_XU : pseudo_unary_w_v;
defm PseudoVFWCVT_F_X  : pseudo_unary_w_v;
defm PseudoVFWCVT_F_F  : pseudo_unary_w_v;

defm PseudoVFNCVT_XU_F : pseudo_unary_v_w;
defm PseudoVFNCVT_X_F  : pseudo_unary_v_w;
defm PseudoVFNCVT_F_XU : pseudo_unary_v_w;
defm PseudoVFNCVT_F_X  : pseudo_unary_v_w;
defm PseudoVFNCVT_F_F  : pseudo_unary_v_w;

defm PseudoVID         : pseudo_nullary_v;

defm PseudoVPOPC       : pseudo_unary_x_m;
defm PseudoVFIRST      : pseudo_unary_x_m;

defm PseudoVMANDNOT    : pseudo_binary_m_mm;
defm PseudoVMAND       : pseudo_binary_m_mm;
defm PseudoVMOR        : pseudo_binary_m_mm;
defm PseudoVMXOR       : pseudo_binary_m_mm;
defm PseudoVMORNOT     : pseudo_binary_m_mm;
defm PseudoVMNAND      : pseudo_binary_m_mm;
defm PseudoVMNOR       : pseudo_binary_m_mm;
defm PseudoVMXNOR      : pseudo_binary_m_mm;

// Load/store pseudo instructions

foreach evr = AllVRegs in
foreach eew = AllEEW in
{
  defvar vlmul = evr.VLMul;

  let mayLoad = 1, mayStore = 0, hasSideEffects = 0,
      usesCustomInserter = 1,
      VLMul = vlmul,
      Constraints = Join<[!if(!gt(vlmul, 1), "@earlyclobber $rd", ""),
                          "$rd = $merge"], ",">.ret in
  {
    let Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
        BaseInstr = !cast<Instruction>("VLE"#eew#"_V") in
      def PseudoVLE#eew#_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, VMaskOp:$mask, GPR:$vl,
                    ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        BaseInstr = !cast<Instruction>("VLSE"#eew#"_V") in
      def PseudoVLSE#eew#_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, GPR:$rs2, VMaskOp:$mask,
                    GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        BaseInstr = !cast<Instruction>("VLXEI"#eew#"_V") in
      def PseudoVLXEI#eew#_V_M#vlmul
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, evr:$rs2, VMaskOp:$mask,
                    GPR:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }

  let mayLoad = 0, mayStore = 1, hasSideEffects = 0,
      usesCustomInserter = 1,
      VLMul = vlmul in
  {
    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        BaseInstr = !cast<Instruction>("VSE"#eew#"_V") in
      def PseudoVSE#eew#_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, VMaskOp:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        BaseInstr = !cast<Instruction>("VSSE"#eew#"_V") in
      def PseudoVSSE#eew#_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, GPR:$rs2, VMaskOp:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        BaseInstr = !cast<Instruction>("VSXEI"#eew#"_V") in
      def PseudoVSXEI#eew#_V_M#vlmul
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, evr:$rs2, VMaskOp:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }
}

//===----------------------------------------------------------------------===//
// Pseudo instructions we need for COPY LMUL>1
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1,
    MergeOpIndex = -1, VLIndex = -1, SEWIndex = -1 in
{
  let BaseInstr = VMV2R_V, VLMul = 2 in
  def PseudoVMV2R_M2 : Pseudo<(outs VRM2:$rdest), (ins VRM2:$rs1), []>,
                             EPIPseudo;

  let BaseInstr = VMV4R_V, VLMul = 4 in
  def PseudoVMV4R_M4 : Pseudo<(outs VRM4:$rdest), (ins VRM4:$rs1), []>,
                             EPIPseudo;

  let BaseInstr = VMV8R_V, VLMul = 8 in
  def PseudoVMV8R_M8 : Pseudo<(outs VRM8:$rdest), (ins VRM8:$rs1), []>,
                             EPIPseudo;
}

}

let Predicates = [HasStdExtV, HasStdExtZvlsseg] in {

// FIXME: Generalize for LMUL>1

foreach eew = AllEEW in
{
  let mayLoad = 1, mayStore = 0, hasSideEffects = 0,
      usesCustomInserter = 1, VLMul = 1 in
  {
    let Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
        BaseInstr = !cast<Instruction>("VLSEG2E"#eew#"_V") in
      def PseudoVLSEG2E#eew#_V_M1
          : Pseudo<(outs VRTuple:$rd),
                   (ins VRTuple:$merge, GPR:$rs1, VMaskOp:$mask, GPR:$vl,
                    ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }

  let mayLoad = 0, mayStore = 1, hasSideEffects = 0,
      usesCustomInserter = 1, VLMul = 1 in
  {
    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        BaseInstr = !cast<Instruction>("VSSEG2E"#eew#"_V") in
      def PseudoVSSEG2E#eew#_V_M1
          : Pseudo<(outs),
                   (ins VRTuple:$rd, GPR:$rs1, VMaskOp:$mask, GPR:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }
}

}

//===----------------------------------------------------------------------===//
// Patterns. Essential
//===----------------------------------------------------------------------===//

// Note: PatLeaf can't be used here as it is considered an input pattern.
def VLMax : OutPatFrag<(ops), (i64 X0)>;

// Floating point instructions with a scalar operand expect such operand to be
// in a register of class FPR64. When dealing with the f32 variant of such
// instructions we need to insert the FPR32 subregister into the FPR64 base
// register to match the instruction operand
class ToFPR64<DAGOperand operand, dag input_dag> {
  dag ret = !if(!eq(!cast<string>(operand),
                    !cast<string>(FPR32)),
                (INSERT_SUBREG (IMPLICIT_DEF), input_dag, sub_32),
                input_dag);
}

// Floating point instructions with a scalar result will generate such result
// in a register of class FPR64. When dealing with the f32 variant of a pattern
// we need to demote the FPR64 base register generated by the instruction to
// the FPR32 subregister expected by the type in the pattern
class FromFPR64<DAGOperand operand, dag input_dag> {
  dag ret = !if(!eq(!cast<string>(operand),
                    !cast<string>(FPR32)),
                (f32 (EXTRACT_SUBREG
                      input_dag,
                      sub_32)),
                input_dag);
}

class FPZero<DAGOperand operand> {
  dag ret = !if(!eq(!cast<string>(operand),
                    !cast<string>(FPR64)),
                (FMV_D_X X0), (FMV_W_X X0));
}

multiclass pat_load_store<LLVMType type,
                          LLVMType mask_type,
                          int eew,
                          int vlmul,
                          VReg reg_class>
{
  // Load
  defvar load_instr_name = "PseudoVLE"#eew#"_V";
  def : Pat<(type (load GPR:$rs1)),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             (type (IMPLICIT_DEF)),
             GPR:$rs1,
             (mask_type zero_reg),
             VLMax, eew)>;
  def : Pat<(type (load AddrFI:$rs1)),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             (type (IMPLICIT_DEF)),
             AddrFI:$rs1,
             (mask_type zero_reg),
             VLMax, eew)>;

  // Store
  defvar store_instr_name = "PseudoVSE"#eew#"_V";
  def : Pat<(store type:$rs2, GPR:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, GPR:$rs1,
             (mask_type zero_reg),
             VLMax, eew)>;
  def : Pat<(store type:$rs2, AddrFI:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, AddrFI:$rs1,
             (mask_type zero_reg),
             VLMax, eew)>;
}

// Extra patterns for truncating loads and extending stores of mask types
multiclass pat_load_store_mask<LLVMType type,
                               int eew,
                               int vlmul,
                               VReg reg_class>
{
  // Load
  defvar load_instr_name = "PseudoVLE"#eew#"_V";
  def : Pat<(type (load GPR:$rs1)),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             (type (IMPLICIT_DEF)),
             GPR:$rs1,
             (type zero_reg),
             VLMax, eew)>;
  def : Pat<(type (load AddrFI:$rs1)),
            (!cast<Instruction>(load_instr_name#"_M"#vlmul)
             (type (IMPLICIT_DEF)),
             AddrFI:$rs1,
             (type zero_reg),
             VLMax, eew)>;

  // Store
  defvar store_instr_name = "PseudoVSE"#eew#"_V";
  def : Pat<(store type:$rs2, GPR:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, GPR:$rs1,
             (type zero_reg),
             VLMax, eew)>;
  def : Pat<(store type:$rs2, AddrFI:$rs1),
            (!cast<Instruction>(store_instr_name#"_M"#vlmul)
             reg_class:$rs2, AddrFI:$rs1,
             (type zero_reg),
             VLMax, eew)>;
}

let Predicates = [HasStdExtV] in {
  foreach vti = AllVectors in
  {
    defm : pat_load_store<vti.Vector, vti.Mask, vti.SEW, vti.RegClass.VLMul,
                          vti.RegClass>;
  }

  foreach mti = AllMasks in
  {
    // Mask values are stored as vectors of bytes
    defm : pat_load_store_mask<mti.Mask, /* eew */ 8, /* vlmul */ 1, VR>;
  }

  foreach vti = AllIntegerVectors in {
    def : Pat<(vti.Vector (zext (vti.Mask V0))),
              (!cast<Instruction>("PseudoVMERGE_VIM_M"#vti.RegClass.VLMul)
               (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul) 0,
                VLMax, vti.SEW),
               1,
               (vti.Mask V0),
               VLMax, vti.SEW)>;
    // No way to do 'anyext' for masks, so we do 'zext' instead.
    def : Pat<(vti.Vector (anyext (vti.Mask V0))),
              (!cast<Instruction>("PseudoVMERGE_VIM_M"#vti.RegClass.VLMul)
               (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul) 0,
                VLMax, vti.SEW),
               1,
               (vti.Mask V0),
               VLMax, vti.SEW)>;
    def : Pat<(vti.Mask (trunc (vti.Vector vti.RegClass:$rs1))),
              (!cast<Instruction>("PseudoVMSNE_VV_M"#vti.RegClass.VLMul)
               (vti.Mask (IMPLICIT_DEF)),
               (!cast<Instruction>("PseudoVAND_VI_M"#vti.RegClass.VLMul)
                (vti.Vector (IMPLICIT_DEF)),
                (vti.Vector vti.RegClass:$rs1), 1,
                (vti.Mask zero_reg),
                VLMax, vti.SEW),
               (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul) 0,
                VLMax, vti.SEW),
               (vti.Mask zero_reg),
               VLMax, vti.SEW)>;
  }
}

//===----------------------------------------------------------------------===//
// Patterns. Common
//===----------------------------------------------------------------------===//

// Using the X0 register or a GPR with value 0 as operands for VSETVL[I] have
// different semantincs: X0 entails VL := VLMAX, while using a GPR with value 0
// entails VL := 0. Uses of a i64 0 constant may be selected into X0, and thus
// the following node is used to represent a register or constant that may hold
// a 0 value and shouldn't be selected into X0.
//
// Note: 'undef' is used as Opcode parameter, given that there is no 'wildcard'
// SDNode.
def NoX0 : SDNodeXForm<undef,
[{
    SDLoc DL(N);

    if (auto *C = dyn_cast<ConstantSDNode>(N)) {
      if (C->isNullValue()) {
        return SDValue(CurDAG->getMachineNode(RISCV::ADDI, DL, MVT::i64,
                           CurDAG->getRegister(RISCV::X0, MVT::i64),
                           CurDAG->getTargetConstant(0, DL, MVT::i64)), 0);
      }
    }

    return SDValue(N, 0);
}]>;

def pow2uimm5 : ImmLeaf<XLenVT, [{
    return isPowerOf2_64(Imm) && isUInt<5>(Log2_64(Imm));
}]>;
def Log2 : SDNodeXForm<imm,
[{
    return CurDAG->getTargetConstant(Log2_64(N->getZExtValue()), SDLoc(N),
        MVT::i64);
}]>;

let Predicates = [HasStdExtV] in {

def : Pat<(int_vscale), (PseudoVSCALE)>;

// FIXME: These patterns can be improved.
def : Pat<(vscale 1), (PseudoVSCALE)>;
def : Pat<(vscale -1), (SUB X0, (PseudoVSCALE))>;
def : Pat<(vscale pow2uimm5:$mul),
          (SLLI (PseudoVSCALE), (Log2 imm:$mul))>;
def : Pat<(vscale (i64 GPR:$imm)),
          (MUL (PseudoVSCALE), $imm)>;

}

//===----------------------------------------------------------------------===//
// Patterns. Arithmetic
//===----------------------------------------------------------------------===//

class swap_helper<dag Prefix,
                  dag A,
                  dag B,
                  dag Suffix,
                  bit swap> {
   dag Value = !con(
       Prefix,
       !if(swap, B, A),
       !if(swap, A, B),
       Suffix);
}

multiclass pat_intrinsic_binary<string intrinsic_name,
                                string instruction_name,
                                string kind,
                                ValueType result_type,
                                ValueType op1_type,
                                ValueType op2_type,
                                ValueType mask_type,
                                int sew,
                                int vlmul,
                                VReg result_reg_class,
                                VReg op1_reg_class,
                                DAGOperand op2_kind,
                                bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name#_#kind#"_M"#vlmul);

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_kind:$rs2),
                          (i64 GPR:$vl))),
            swap_helper<
              (instruction (result_type (IMPLICIT_DEF))),
              (instruction
               (op1_type op1_reg_class:$rs1)),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction
               (mask_type zero_reg),
               (NoX0 GPR:$vl),
               sew),
              swap>.Value>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$merge),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_kind:$rs2),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            swap_helper<
              (instruction result_reg_class:$merge),
              (instruction
               op1_reg_class:$rs1),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction
               (mask_type V0),
               (NoX0 GPR:$vl),
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary_nomask<string intrinsic_name,
                                       string instruction_name,
                                       string kind,
                                       ValueType result_type,
                                       ValueType op1_type,
                                       ValueType op2_type,
                                       ValueType mask_type,
                                       int sew,
                                       int vlmul,
                                       VReg op1_reg_class,
                                       DAGOperand op2_kind,
                                       bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name#_#kind#"_M"#vlmul);

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_kind:$rs2),
                          (i64 GPR:$vl))),
            swap_helper<
              (instruction), // empty prefix
              (instruction
               (op1_type op1_reg_class:$rs1)),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction
               (NoX0 GPR:$vl),
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary_mask_in<string intrinsic_name,
                                        string instruction_name,
                                        string kind,
                                        ValueType result_type,
                                        ValueType op1_type,
                                        ValueType op2_type,
                                        ValueType mask_type,
                                        int sew,
                                        int vlmul,
                                        VReg result_reg_class,
                                        VReg op1_reg_class,
                                        DAGOperand op2_kind,
                                        bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name#_#kind#"_M"#vlmul);

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_kind:$rs2),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            swap_helper<
              (instruction), // empty prefix
              (instruction
               op1_reg_class:$rs1),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction
               (mask_type V0),
               (NoX0 GPR:$vl),
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary_int_v_vv<string intrinsic_name,
                                         string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_int_v_vv_vx<string intrinsic_name,
                                            string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_v_vx_vi<string intrinsic_name,
                                            string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_v_vv_vx_vi<string intrinsic_name,
                                               string instruction_name,
                                               DAGOperand imm_kind = simm5>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, imm_kind>;
  }
}

multiclass pat_intrinsic_binary_int_v_vvm_vxm<string intrinsic_name,
                                              string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Vector, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass,
                                        vti.RegClass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Vector, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_v_vvm_vxm_vim<string intrinsic_name,
                                                  string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Vector, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass,
                                        vti.RegClass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Vector, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VIM",
                                        vti.Vector, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_m_vvm_vxm<string intrinsic_name,
                                              string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Mask, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass,
                                        vti.RegClass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Mask, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_m_vvm_vxm_vim<string intrinsic_name,
                                                  string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Mask, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass,
                                        vti.RegClass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Mask, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VIM",
                                        vti.Mask, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                        vti.RegClass, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_vx_nomask<string intrinsic_name,
                                                   string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VV",
                                       vti.Mask, vti.Vector, vti.Vector,
                                       vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                       vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VX",
                                       vti.Mask, vti.Vector, XLenVT,
                                       vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                       vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_vx_vi_nomask<string intrinsic_name,
                                                      string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VV",
                                       vti.Mask, vti.Vector, vti.Vector,
                                       vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                       vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VX",
                                       vti.Mask, vti.Vector, XLenVT,
                                       vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                       vti.RegClass, GPR>;
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VI",
                                       vti.Mask, vti.Vector, XLenVT,
                                       vti.Mask, vti.SEW, vti.RegClass.VLMul,
                                       vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_v_vs<string intrinsic_name,
                                         string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VS",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_int_w_vv_vx<string intrinsic_name,
                                            string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                wti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, wti.RegClass,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                wti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, wti.RegClass,
                                vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_w_wv_wx<string intrinsic_name,
                                            string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WV",
                                wti.Vector, wti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, wti.RegClass,
                                wti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WX",
                                wti.Vector, wti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, wti.RegClass,
                                wti.RegClass, GPR>;
  }
}

// This is the narrowing form.
multiclass pat_intrinsic_binary_int_v_wv_wx_wi<string intrinsic_name,
                                               string instruction_name,
                                               DAGOperand imm_kind = simm5>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WV",
                                vti.Vector, wti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                wti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WX",
                                vti.Vector, wti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                wti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WI",
                                vti.Vector, wti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                wti.RegClass, imm_kind>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_vx_vi<string intrinsic_name,
                                               string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, VR,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_vx<string intrinsic_name,
                                           string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, VR,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_m_vx_vi<string intrinsic_name,
                                            string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.RegClass.VLMul, VR, vti.RegClass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_swapped<string intrinsic_name,
                                                 string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, VR,
                                vti.RegClass, vti.RegClass, /* swapped */ 1>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vv<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass, fvti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vf<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vs<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VS",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass, fvti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vv_vf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fvti.RegClass, fvti.RegClass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vvm<string intrinsic_name,
                                         string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        fvti.Vector, fvti.Vector, fvti.Vector,
                                        fvti.Mask, fvti.SEW,
                                        fvti.RegClass.VLMul, fvti.RegClass,
                                        fvti.RegClass, fvti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vfm<string intrinsic_name,
                                         string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VFM",
                                        fvti.Vector, fvti.Vector, fvti.Scalar,
                                        fvti.Mask, fvti.SEW,
                                        fvti.RegClass.VLMul, fvti.RegClass,
                                        fvti.RegClass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_w_vv_vf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fwti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fwti.RegClass, fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fwti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fwti.RegClass, fvti.RegClass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_w_wv_wf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WV",
                                fwti.Vector, fwti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fwti.RegClass, fwti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WF",
                                fwti.Vector, fwti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                fwti.RegClass, fwti.RegClass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_m_vv_vf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Mask, fvti.Vector, fvti.Vector, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, VR,
                                fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Mask, fvti.Vector, fvti.Scalar, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, VR,
                                fvti.RegClass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_m_vf<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Mask, fvti.Vector, fvti.Scalar, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, VR,
                                fvti.RegClass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_m_vv_swapped<string intrinsic_name,
                                                string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Mask, fvti.Vector, fvti.Vector, fvti.Mask,
                                fvti.SEW, fvti.RegClass.VLMul, VR,
                                fvti.RegClass, fvti.RegClass, /* swapped */ 1>;
  }
}

multiclass pat_intrinsic_binary_m_mm<string intrinsic_name,
                                     string instruction_name>
{
  foreach mti = AllMasks in
  {
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "MM",
                                       mti.Mask, mti.Mask, mti.Mask, mti.Mask,
                                       mti.SEW, mti.VLMul, VR, VR>;
  }
}

multiclass pat_intrinsic_ternary<string intrinsic_name,
                                 string instruction_name,
                                 string kind,
                                 ValueType result_type,
                                 ValueType op1_type,
                                 ValueType op2_type,
                                 ValueType mask_type,
                                 int sew,
                                 int vlmul,
                                 VReg result_reg_class,
                                 RegisterClass op1_reg_class,
                                 VReg op2_reg_class>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             result_reg_class:$rs3,
             ToFPR64<op1_reg_class, (op1_type op1_reg_class:$rs1)>.ret,
             op2_reg_class:$rs2,
             (mask_type zero_reg),
             (NoX0 GPR:$vl),
             sew)>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             result_reg_class:$rs3,
             ToFPR64<op1_reg_class, (op1_type op1_reg_class:$rs1)>.ret,
             op2_reg_class:$rs2,
             (mask_type V0),
             (NoX0 GPR:$vl),
             sew)>;
}

multiclass pat_intrinsic_ternary_int_v_vv_vx<string intrinsic_name,
                                             string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VV",
                                 vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                 vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                 vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VX",
                                 vti.Vector, XLenVT, vti.Vector, vti.Mask,
                                 vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                 GPR, vti.RegClass>;
  }
}

multiclass pat_intrinsic_ternary_fp_v_vv_vf<string intrinsic_name,
                                            string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VV",
                                 fvti.Vector, fvti.Vector, fvti.Vector,
                                 fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                 fvti.RegClass, fvti.RegClass, fvti.RegClass>;
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VF",
                                 fvti.Vector, fvti.Scalar, fvti.Vector,
                                 fvti.Mask, fvti.SEW, fvti.RegClass.VLMul,
                                 fvti.RegClass, fvti.ScalarRegClass,
                                 fvti.RegClass>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vx<string intrinsic_name,
                                                 string instruction_name>
{
  foreach vti = AllVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vv_vx_vi<string intrinsic_name,
                                                       string instruction_name,
                                                       DAGOperand imm_kind = simm5>
{
  foreach vti = AllVectors in
  {
    defvar ivti = GetIntVectorTypeInfo<vti>.Vti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, ivti.Vector, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, vti.RegClass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, imm_kind>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vx_vi<string intrinsic_name,
                                                    string instruction_name,
                                                    DAGOperand imm_kind = simm5>
{
  foreach vti = AllVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.RegClass.VLMul, vti.RegClass,
                                vti.RegClass, imm_kind>;
  }
}

//class ValueTypePair<ValueType ty1, ValueType ty2>
//{
//   ValueType First = ty1;
//   ValueType Second = ty2;
//}

//class ValueTypeVarList<list<ValueTypePair> VList>
//{
//  list<ValueTypePair> Value = VList;
//}

//def SameSizePairs : ValueTypeVarList<
//  [ ValueTypePair<nxv1i32, nxv1f32>,
//    ValueTypePair<nxv1i64, nxv1f64> ]>;

//multiclass pat_conversions<string instruction,
//                           string intrinsic,
//                           list<ValueTypePair> pairs>
//{

//foreach vtp = pairs in
//{
//def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic) (vtp.Second VR:$rs1))),
//          (!cast<Instruction>(instruction # "_V") VR:$rs1)>;
//def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic # "_mask") (vtp.Second VR:$rs1), V0)),
//          (!cast<Instruction>(instruction # "_V_MASK") VR:$rs1, vmask_only_true.Value)>;

//}

//}

multiclass pat_intrinsic_unary<string intrinsic_name,
                               string instruction_name,
                               string kind,
                               ValueType result_type,
                               ValueType op1_type,
                               ValueType mask_type,
                               int sew,
                               int vlmul,
                               VReg result_reg_class,
                               VReg op1_reg_class>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             (result_type (IMPLICIT_DEF)),
             op1_reg_class:$rs1,
             (mask_type zero_reg),
             (NoX0 GPR:$vl), sew)>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$merge),
                          (op1_type op1_reg_class:$rs1),
                          (mask_type V0),
                          (i64 GPR:$vl))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             result_reg_class:$merge,
             op1_reg_class:$rs1,
             (mask_type V0),
             (NoX0 GPR:$vl), sew)>;
}

multiclass pat_intrinsic_unary_int_w_v<string intrinsic_name,
                                       string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               wti.Vector, vti.Vector, wti.Mask, vti.SEW,
                               vti.RegClass.VLMul, wti.RegClass, vti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_int_v_w<string intrinsic_name,
                                       string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               vti.Vector, wti.Vector, vti.Mask, vti.SEW,
                               vti.RegClass.VLMul, vti.RegClass, wti.RegClass>;
  }
}


multiclass pat_intrinsic_unary_fp_w_v<string intrinsic_name,
                                      string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               fwti.Vector, fvti.Vector, fwti.Mask, fvti.SEW,
                               fvti.RegClass.VLMul, fwti.RegClass,
                               fvti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_int_w_fp_v<string intrinsic_name,
                                          string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar iwti = GetIntVectorTypeInfo<fvtiToFWti.FWti>.Vti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               iwti.Vector, fvti.Vector, iwti.Mask, fvti.SEW,
                               fvti.RegClass.VLMul, iwti.RegClass,
                               fvti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_fp_w_int_v<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntToFloatVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar fwti = GetFloatVectorTypeInfo<vtiToWti.Wti>.FVti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               fwti.Vector, vti.Vector, fwti.Mask, vti.SEW,
                               vti.RegClass.VLMul, fwti.RegClass, vti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_fp_v_w<string intrinsic_name,
                                      string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
                               fvti.Vector, fwti.Vector, fvti.Mask, fvti.SEW,
                               fvti.RegClass.VLMul, fvti.RegClass,
                               fwti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_int_v_fp_w<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntToFloatVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar fwti = GetFloatVectorTypeInfo<vtiToWti.Wti>.FVti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
                               vti.Vector, fwti.Vector, vti.Mask, vti.SEW,
                               vti.RegClass.VLMul, vti.RegClass, fwti.RegClass>;
  }
}

multiclass pat_intrinsic_unary_fp_v_int_w<string intrinsic_name,
                                          string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar iwti = GetIntVectorTypeInfo<fvtiToFWti.FWti>.Vti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
                               fvti.Vector, iwti.Vector, fvti.Mask, fvti.SEW,
                               fvti.RegClass.VLMul, fvti.RegClass,
                               iwti.RegClass>;
  }
}

//// We are missing cases but are of sizes that won't work
//def WidenedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1i64, nxv1f32>,
//    ValueTypePair<nxv1f32, nxv1i16>,
//    ValueTypePair<nxv1f64, nxv1i32> ]>;

//// Narrowing
//// TODO: This list is the inverse of the one above
//def NarrowedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f32, nxv1i64>,
//    ValueTypePair<nxv1i16, nxv1f32>,
//    ValueTypePair<nxv1i32, nxv1f64> ]>;

//def FloatWidenedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f64, nxv1f32> ]>;

//def FloatNarrowedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f32, nxv1f64> ]>;

let Predicates = [HasStdExtV] in {

defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vadd", "PseudoVADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vsub", "PseudoVSUB">;
defm "" : pat_intrinsic_binary_int_v_vx_vi<"int_epi_vrsub", "PseudoVRSUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vminu", "PseudoVMINU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmin", "PseudoVMIN">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmaxu", "PseudoVMAXU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmax", "PseudoVMAX">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vand", "PseudoVAND">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vor", "PseudoVOR">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vxor", "PseudoVXOR">;

defm "" : pat_intrinsic_binary_any_and_int_v_vv_vx_vi<"int_epi_vrgather", "PseudoVRGATHER", uimm5>;
defm "" : pat_intrinsic_binary_any_and_int_v_vx_vi<"int_epi_vslideup", "PseudoVSLIDEUP", uimm5>;
defm "" : pat_intrinsic_binary_any_and_int_v_vx_vi<"int_epi_vslidedown", "PseudoVSLIDEDOWN", uimm5>;

defm "" : pat_intrinsic_binary_int_v_vvm_vxm_vim<"int_epi_vadc", "PseudoVADC">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_vi_nomask<"int_epi_vmadc", "PseudoVMADC">;
defm "" : pat_intrinsic_binary_int_m_vvm_vxm_vim<"int_epi_vmadc_carry_in", "PseudoVMADC">;

defm "" : pat_intrinsic_binary_int_v_vvm_vxm<"int_epi_vsbc", "PseudoVSBC">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_nomask<"int_epi_vmsbc", "PseudoVMSBC">;
defm "" : pat_intrinsic_binary_int_m_vvm_vxm<"int_epi_vmsbc_borrow_in", "PseudoVMSBC">;

defm "" : pat_intrinsic_binary_int_m_vv_vx_vi<"int_epi_vmseq", "PseudoVMSEQ">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_vi<"int_epi_vmsne", "PseudoVMSNE">;
defm "" : pat_intrinsic_binary_int_m_vv_vx<"int_epi_vmsltu", "PseudoVMSLTU">;
defm "" : pat_intrinsic_binary_int_m_vv_vx<"int_epi_vmslt", "PseudoVMSLT">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_vi<"int_epi_vmsleu", "PseudoVMSLEU">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_vi<"int_epi_vmsle", "PseudoVMSLE">;

defm "" : pat_intrinsic_binary_int_m_vx_vi<"int_epi_vmsgtu", "PseudoVMSGTU">;
// Select (int_epi_vmsgtu reg:$rs1, reg:$rs2) as (PseudoVMSLTU reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_int_m_vv_swapped<"int_epi_vmsgtu", "PseudoVMSLTU">;

defm "" : pat_intrinsic_binary_int_m_vx_vi<"int_epi_vmsgt", "PseudoVMSGT">;
// Select (int_epi_vmsgt reg:$rs1, reg:$rs2) as (PseudoVMSLT reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_int_m_vv_swapped<"int_epi_vmsgt", "PseudoVMSLT">;

defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsaddu", "PseudoVSADDU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsadd", "PseudoVSADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vssubu", "PseudoVSSUBU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vssub", "PseudoVSSUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vaadd", "PseudoVAADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsll", "PseudoVSLL", uimm5>;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vasub", "PseudoVASUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vsmul", "PseudoVSMUL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsrl", "PseudoVSRL", uimm5>;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsra", "PseudoVSRA", uimm5>;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vssrl", "PseudoVSSRL", uimm5>;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vssra", "PseudoVSSRA", uimm5>;

defm "" : pat_intrinsic_binary_int_v_vvm_vxm_vim<"int_epi_vmerge", "PseudoVMERGE">;

defm "" : pat_intrinsic_binary_int_v_wv_wx_wi<"int_epi_vnsrl", "PseudoVNSRL", uimm5>;
defm "" : pat_intrinsic_binary_int_v_wv_wx_wi<"int_epi_vnsra", "PseudoVNSRA", uimm5>;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnclipu", "VNCLIPU", uimm5>;
//defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnclip", "VNCLIP", uimm5>;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vs<"int_epi_vwredsumu", "VWREDSUMU">;
//defm "" : pat_intrinsic_binary_int_vs<"int_epi_vwredsum", "VWREDSUM">;

// FIXME EDIV patterns disabled
//defm "" : pat_intrinsic_binary_int_v_vv<"int_epi_vdotu", "PseudoVDOTU">;
//defm "" : pat_intrinsic_binary_int_v_vv<"int_epi_vdot", "PseudoVDOT">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmaccu", "VWSMACCU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmacc", "VWSMACC">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmsacu", "VWSMSACU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmsac", "VWSMSAC">;

defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredsum", "PseudoVREDSUM">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredand", "PseudoVREDAND">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredor", "PseudoVREDOR">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredxor", "PseudoVREDXOR">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredminu", "PseudoVREDMINU">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmin", "PseudoVREDMIN">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmaxu", "PseudoVREDMAXU">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmax", "PseudoVREDMAX">;

// The 'int.epi.vmv_x_s' intrinsic will have an illegal result type (i8, i16,
// i32) whenever the operand type is nxv8i8, nxv4i16 or nxv2i32. In such cases,
// the intrinsic is legalized programatically using the RISCVISD::VMV_X_S custom
// node, which is then selected using a pattern as usual.
foreach vti = [Vtype1xi64] in
  def : Pat<(int_epi_vmv_x_s (vti.Vector vti.RegClass:$rs2)),
            (PseudoVMV_X_S $rs2, vti.SEW)>;
foreach vti = [Vtype2xi32, Vtype4xi16, Vtype8xi8] in
  def : Pat<(riscv_vmv_x_s (vti.Vector vti.RegClass:$rs2)),
            (PseudoVMV_X_S $rs2, vti.SEW)>;

foreach vti = NoGroupIntegerVectors in
  def : Pat<(vti.Vector (int_epi_vmv_s_x (vti.Vector VR:$merge), GPR:$rs1, GPR:$vl)),
            (PseudoVMV_S_X $merge, $rs1, (NoX0 GPR:$vl), vti.SEW)>;

defm "" : pat_intrinsic_binary_any_and_int_v_vx<"int_epi_vslide1up", "PseudoVSLIDE1UP">;
defm "" : pat_intrinsic_binary_any_and_int_v_vx<"int_epi_vslide1down", "PseudoVSLIDE1DOWN">;

foreach mti = AllMasks in {
  def : Pat<(int_epi_vpopc (mti.Mask VR:$rs2), GPR:$vl),
            (!cast<Instruction>("PseudoVPOPC_M_M"#mti.VLMul) $rs2,
              (mti.Mask zero_reg), (NoX0 GPR:$vl), mti.SEW)>;
  def : Pat<(int_epi_vpopc_mask (mti.Mask VR:$rs2), (mti.Mask V0), GPR:$vl),
            (!cast<Instruction>("PseudoVPOPC_M_M"#mti.VLMul) $rs2,
              (mti.Mask V0), (NoX0 GPR:$vl), mti.SEW)>;

  def : Pat<(int_epi_vfirst (mti.Mask VR:$rs2), GPR:$vl),
            (!cast<Instruction>("PseudoVFIRST_M_M"#mti.VLMul) $rs2,
              (mti.Mask zero_reg), (NoX0 GPR:$vl), mti.SEW)>;
  def : Pat<(int_epi_vfirst_mask (mti.Mask VR:$rs2), (mti.Mask V0), GPR:$vl),
            (!cast<Instruction>("PseudoVFIRST_M_M"#mti.VLMul) $rs2,
              (mti.Mask V0), (NoX0 GPR:$vl), mti.SEW)>;
}

//def : Pat<(int_epi_vmsbf VR:$rs1),
//        (VMSBF_M $rs1)>;
//def : Pat<(int_epi_vmsbf_mask VR:$rs1, V0),
//        (VMSBF_M_MASK $rs1, vmask_only_true.Value)>;

//def : Pat<(int_epi_vmsif VR:$rs1),
//        (VMSIF_M $rs1)>;
//def : Pat<(int_epi_vmsif_mask VR:$rs1, V0),
//        (VMSIF_M_MASK $rs1, vmask_only_true.Value)>;

//def : Pat<(int_epi_vmsof VR:$rs1),
//        (VMSOF_M $rs1)>;
//def : Pat<(int_epi_vmsof_mask VR:$rs1, V0),
//        (VMSOF_M_MASK $rs1, vmask_only_true.Value)>;

//foreach vtp = AllVectors in
//def : Pat<(vtp.Vector (int_epi_vcompress (vtp.Vector VR:$rs2), (nxv1i1 VR:$rs1))),
//          (VCOMPRESS_VM $rs2, $rs1)>;

defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmandnot", "PseudoVMANDNOT">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmand", "PseudoVMAND">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmor", "PseudoVMOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmxor", "PseudoVMXOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmornot", "PseudoVMORNOT">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmnand", "PseudoVMNAND">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmnor", "PseudoVMNOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmxnor", "PseudoVMXNOR">;

defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vdivu", "PseudoVDIVU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vdiv", "PseudoVDIV">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vremu", "PseudoVREMU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vrem", "PseudoVREM">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulhu", "PseudoVMULHU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmul", "PseudoVMUL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulhsu", "PseudoVMULHSU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulh", "PseudoVMULH">;

defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmadd", "PseudoVMADD">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vnmsub", "PseudoVNMSUB">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmacc", "PseudoVMACC">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vnmsac", "PseudoVNMSAC">;

defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwaddu", "PseudoVWADDU">;
defm "" : pat_intrinsic_binary_int_w_wv_wx<"int_epi_vwaddu_w", "PseudoVWADDU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwadd", "PseudoVWADD">;
defm "" : pat_intrinsic_binary_int_w_wv_wx<"int_epi_vwadd_w", "PseudoVWADD">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwsubu", "PseudoVWSUBU">;
defm "" : pat_intrinsic_binary_int_w_wv_wx<"int_epi_vwsubu_w", "PseudoVWSUBU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwsub", "PseudoVWSUB">;
defm "" : pat_intrinsic_binary_int_w_wv_wx<"int_epi_vwsub_w", "PseudoVWSUB">;

defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmulu", "PseudoVWMULU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmulsu", "PseudoVWMULSU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmul", "PseudoVWMUL">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmaccu", "VWMACCU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmacc", "VWMACC">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmsacu", "VWMSACU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmsac", "VWMSAC">;

defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfadd", "PseudoVFADD">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredsum", "PseudoVFREDSUM">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsub", "PseudoVFSUB">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredosum", "PseudoVFREDOSUM">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmin", "PseudoVFMIN">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredmin", "PseudoVFREDMIN">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmax", "PseudoVFMAX">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredmax", "PseudoVFREDMAX">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnj", "PseudoVFSGNJ">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnjn", "PseudoVFSGNJN">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnjx", "PseudoVFSGNJX">;

foreach fvti = NoGroupFloatVectors in
{
  def : Pat<(fvti.Scalar (int_epi_vfmv_f_s (fvti.Vector fvti.RegClass:$rs2))),
             FromFPR64<fvti.ScalarRegClass, (PseudoVFMV_F_S $rs2, fvti.SEW)>.ret>;

  def : Pat<(fvti.Vector (int_epi_vfmv_s_f (fvti.Vector VR:$merge),
                          (fvti.Scalar fvti.ScalarRegClass:$rs2), GPR:$vl)),
            (PseudoVFMV_S_F
             $merge,
             ToFPR64<fvti.ScalarRegClass,
                     (fvti.Scalar fvti.ScalarRegClass:$rs2)>.ret,
             (NoX0 GPR:$vl), fvti.SEW)>;
}

defm "" : pat_intrinsic_binary_fp_m_vv_vf<"int_epi_vmfeq", "PseudoVMFEQ">;
defm "" : pat_intrinsic_binary_fp_m_vv_vf<"int_epi_vmfle", "PseudoVMFLE">;
defm "" : pat_intrinsic_binary_fp_m_vv_vf<"int_epi_vmflt", "PseudoVMFLT">;
defm "" : pat_intrinsic_binary_fp_m_vv_vf<"int_epi_vmfne", "PseudoVMFNE">;

defm "" : pat_intrinsic_binary_fp_m_vf<"int_epi_vmfgt", "PseudoVMFGT">;
// Select (int_epi_vmfgt reg:$rs1, reg:$rs2) as (PseudoVMFLT reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_fp_m_vv_swapped<"int_epi_vmfgt", "PseudoVMFLT">;

defm "" : pat_intrinsic_binary_fp_m_vf<"int_epi_vmfge", "PseudoVMFGE">;
// Select (int_epi_vmfge reg:$rs1, reg:$rs2) as (PseudoVMFLE reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_fp_m_vv_swapped<"int_epi_vmfge", "PseudoVMFLE">;

defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfdiv", "PseudoVFDIV">;
defm "" : pat_intrinsic_binary_fp_v_vf<"int_epi_vfrdiv", "PseudoVFRDIV">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmul", "PseudoVFMUL">;

// There is no "vfmerge.vvm" so we select vmerge.vvm
defm "" : pat_intrinsic_binary_fp_v_vvm<"int_epi_vfmerge", "PseudoVMERGE">;
defm "" : pat_intrinsic_binary_fp_v_vfm<"int_epi_vfmerge", "PseudoVFMERGE">;

defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmadd", "PseudoVFMADD">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmadd", "PseudoVFNMADD">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmsub", "PseudoVFMSUB">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmsub", "PseudoVFNMSUB">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmacc", "PseudoVFMACC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmacc", "PseudoVFNMACC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmsac", "PseudoVFMSAC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmsac", "PseudoVFNMSAC">;

defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwadd", "PseudoVFWADD">;
defm "" : pat_intrinsic_binary_fp_w_wv_wf<"int_epi_vfwadd_w", "PseudoVFWADD">;
//defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfwredsum", "VFWREDSUM">;
defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwsub", "PseudoVFWSUB">;
defm "" : pat_intrinsic_binary_fp_w_wv_wf<"int_epi_vfwsub_w", "PseudoVFWSUB">;
//defm "" : pat_intrinsic_binary_fp_vs<"int_epi_vfwredosum", "VFWREDOSUM">;
defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwmul", "PseudoVFWMUL">;
// FIXME EDIV patterns disabled
//defm "" : pat_intrinsic_binary_fp_v_vv<"int_epi_vfdot", "PseudoVFDOT">;

//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwmacc", "VFWMACC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwnmacc", "VFWNMACC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwmsac", "VFWMSAC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwnmsac", "VFWNMSAC">;

foreach vti = AllFloatVectors in
  def : Pat<(vti.Vector (int_epi_vfsqrt (vti.Vector vti.RegClass:$rs2),
                         GPR:$vl)),
            (!cast<Instruction>("PseudoVFSQRT_V_M" # vti.RegClass.VLMul)
             (vti.Vector (IMPLICIT_DEF)),
             $rs2,
             (vti.Mask zero_reg),
             (NoX0 GPR:$vl), vti.SEW)>;

foreach vti = AllFloatVectors in
  def : Pat<(vti.Vector (int_epi_vfsqrt_mask (vti.Vector vti.RegClass:$merge),
                         (vti.Vector vti.RegClass:$rs2), (vti.Mask V0),
                         GPR:$vl)),
            (!cast<Instruction>("PseudoVFSQRT_V_M" # vti.RegClass.VLMul)
             $merge,
             $rs2,
             (vti.Mask V0),
             (NoX0 GPR:$vl), vti.SEW)>;

//foreach itp = [ nxv1i32, nxv1i64 ] in
//foreach vtp = AllFloatVectors in
//def : Pat<(itp (int_epi_vfclass (vtp.Vector VR:$rs2))),
//          (VFCLASS_V $rs2)>;

//foreach itp = [ nxv1i32, nxv1i64 ] in
//foreach vtp = AllFloatVectors in
//def : Pat<(itp (int_epi_vfclass_mask (vtp.Vector VR:$rs2), V0)),
//          (VFCLASS_V_MASK $rs2, vmask_only_true.Value)>;

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_viota (vti.Mask VR:$rs2), GPR:$vl)),
            (!cast<Instruction>("PseudoVIOTA_M_M"#vti.RegClass.VLMul)
             (vti.Vector (IMPLICIT_DEF)),
             $rs2,
             (vti.Mask zero_reg),
             (NoX0 GPR:$vl), vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_viota_mask (vti.Vector vti.RegClass:$merge),
                         (vti.Mask VR:$rs2), (vti.Mask V0), GPR:$vl)),
            (!cast<Instruction>("PseudoVIOTA_M_M"#vti.RegClass.VLMul)
             $merge,
             $rs2,
             (vti.Mask V0),
             (NoX0 GPR:$vl), vti.SEW)>;
}

multiclass pat_conversions_same_sew<string intrinsic_float_to_int,
                                    string instruction_float_to_int,
                                    string intrinsic_int_to_float,
                                    string instruction_int_to_float>
{
  foreach fvti = AllFloatVectors in
  foreach ivti = [GetIntVectorTypeInfo<fvti>.Vti] in
  {
    defvar vlmul = fvti.RegClass.VLMul;

    def : Pat<(ivti.Vector (!cast<Intrinsic>(intrinsic_float_to_int)
                            (fvti.Vector fvti.RegClass:$rs2),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_float_to_int#"_M"#vlmul)
               (ivti.Vector (IMPLICIT_DEF)),
               $rs2,
               (ivti.Mask zero_reg),
               (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(ivti.Vector (!cast<Intrinsic>(intrinsic_float_to_int#"_mask")
                            (ivti.Vector fvti.RegClass:$merge),
                            (fvti.Vector fvti.RegClass:$rs2), (fvti.Mask V0),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_float_to_int#"_M"#vlmul)
               $merge,
               $rs2,
               (fvti.Mask V0),
               (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(fvti.Vector (!cast<Intrinsic>(intrinsic_int_to_float)
                            (ivti.Vector fvti.RegClass:$rs2),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_int_to_float#"_M"#vlmul)
               (fvti.Vector (IMPLICIT_DEF)),
               $rs2,
               (fvti.Mask zero_reg),
               (NoX0 GPR:$vl), fvti.SEW)>;

    def : Pat<(fvti.Vector (!cast<Intrinsic>(intrinsic_int_to_float#"_mask")
                            (fvti.Vector fvti.RegClass:$merge),
                            (ivti.Vector fvti.RegClass:$rs2), (fvti.Mask V0),
                            GPR:$vl)),
              (!cast<Instruction>(instruction_int_to_float#"_M"#vlmul)
               $merge,
               $rs2,
               (fvti.Mask V0),
               (NoX0 GPR:$vl), fvti.SEW)>;
  }
}


defm "" : pat_conversions_same_sew<"int_epi_vfcvt_xu_f", "PseudoVFCVT_XU_F_V",
                                   "int_epi_vfcvt_f_xu", "PseudoVFCVT_F_XU_V">;
defm "" : pat_conversions_same_sew<"int_epi_vfcvt_x_f", "PseudoVFCVT_X_F_V",
                                   "int_epi_vfcvt_f_x", "PseudoVFCVT_F_X_V">;

defm : pat_intrinsic_unary_int_w_fp_v<"int_epi_vfwcvt_xu_f", "PseudoVFWCVT_XU_F">;
defm : pat_intrinsic_unary_int_w_fp_v<"int_epi_vfwcvt_x_f", "PseudoVFWCVT_X_F">;

defm : pat_intrinsic_unary_fp_w_int_v<"int_epi_vfwcvt_f_xu", "PseudoVFWCVT_F_XU">;
defm : pat_intrinsic_unary_fp_w_int_v<"int_epi_vfwcvt_f_x", "PseudoVFWCVT_F_X">;

defm : pat_intrinsic_unary_fp_w_v<"int_epi_vfwcvt_f_f", "PseudoVFWCVT_F_F">;

defm : pat_intrinsic_unary_int_v_fp_w<"int_epi_vfncvt_xu_f", "PseudoVFNCVT_XU_F">;
defm : pat_intrinsic_unary_int_v_fp_w<"int_epi_vfncvt_x_f", "PseudoVFNCVT_X_F">;

defm : pat_intrinsic_unary_fp_v_int_w<"int_epi_vfncvt_f_xu", "PseudoVFNCVT_F_XU">;
defm : pat_intrinsic_unary_fp_v_int_w<"int_epi_vfncvt_f_x", "PseudoVFNCVT_F_X">;

defm : pat_intrinsic_unary_fp_v_w<"int_epi_vfncvt_f_f", "PseudoVFNCVT_F_F">;

multiclass pat_intrinsic_binary_int_w_vX0<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defvar vlmul = vti.RegClass.VLMul;

    def : Pat<(wti.Vector (!cast<Intrinsic>(intrinsic_name)
                           (vti.Vector vti.RegClass:$rs1),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_M"#vlmul)
               (wti.Vector (IMPLICIT_DEF)),
               vti.RegClass:$rs1,
               /* rs2 */ X0,
               (wti.Mask zero_reg),
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(wti.Vector (!cast<Intrinsic>(intrinsic_name#"_mask")
                           (wti.Vector wti.RegClass:$merge),
                           (vti.Vector vti.RegClass:$rs1),
                           (wti.Mask V0),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_M"#vlmul)
               wti.RegClass:$merge,
               vti.RegClass:$rs1,
               /* rs2 */ X0,
               (wti.Mask V0),
               (NoX0 GPR:$vl), vti.SEW)>;
  }
}

multiclass pat_intrinsic_binary_int_v_wX0<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defvar vlmul = vti.RegClass.VLMul;

    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic_name)
                           (wti.Vector wti.RegClass:$rs1),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_M"#vlmul)
               (vti.Vector (IMPLICIT_DEF)),
               wti.RegClass:$rs1,
               /* rs2 */ X0,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic_name#"_mask")
                           (vti.Vector vti.RegClass:$merge),
                           (wti.Vector wti.RegClass:$rs1),
                           (vti.Mask V0),
                           (i64 GPR:$vl))),
              (!cast<Instruction>(instruction_name#"_M"#vlmul)
               vti.RegClass:$merge,
               wti.RegClass:$rs1,
               /* rs2 */ X0,
               (vti.Mask V0),
               (NoX0 GPR:$vl), vti.SEW)>;
  }
}

defm : pat_intrinsic_binary_int_w_vX0<"int_epi_vwcvtu_x_x", "PseudoVWADDU_VX">;
defm : pat_intrinsic_binary_int_w_vX0<"int_epi_vwcvt_x_x", "PseudoVWADD_VX">;

defm : pat_intrinsic_binary_int_v_wX0<"int_epi_vncvt_x_x", "PseudoVNSRL_WX">;

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vid GPR:$vl)),
            (!cast<Instruction>("PseudoVID_V_M"#vti.RegClass.VLMul)
             (vti.Vector (IMPLICIT_DEF)),
             (vti.Mask zero_reg),
             (NoX0 GPR:$vl), vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_vid_mask (vti.Vector vti.RegClass:$merge),
                         (vti.Mask V0), GPR:$vl)),
            (!cast<Instruction>("PseudoVID_V_M"#vti.RegClass.VLMul)
             $merge,
             (vti.Mask V0),
             (NoX0 GPR:$vl), vti.SEW)>;
}

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vmv_v_x GPR:$rs2, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_X_M"#vti.RegClass.VLMul)
             $rs2,
             (NoX0 GPR:$vl), vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_vmv_v_x simm5:$imm5, GPR:$vl)),
            (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul)
             simm5:$imm5,
             (NoX0 GPR:$vl), vti.SEW)>;
}
foreach fvti = AllFloatVectors in {
  def : Pat<(fvti.Vector (int_epi_vfmv_v_f
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), GPR:$vl)),
            (!cast<Instruction>("PseudoVFMV_V_F_M"#fvti.RegClass.VLMul)
             ToFPR64<fvti.ScalarRegClass,
                     (fvti.Scalar fvti.ScalarRegClass:$rs2)>.ret,
             (NoX0 GPR:$vl), fvti.SEW)>;
}
}

//===----------------------------------------------------------------------===//
// Patterns. Full vector arithmetic operations
//===----------------------------------------------------------------------===//

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_binary_v<SDNode vop,
                            string instruction_name,
                            ValueType result_type,
                            ValueType op_type,
                            ValueType mask_type,
                            int sew,
                            int vlmul,
                            VReg op_reg_class,
                            bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name#"_VV_M"#vlmul);
  def : Pat<(result_type (vop
                          (op_type op_reg_class:$rs1),
                          (op_type op_reg_class:$rs2))),
            swap_helper<
              (instruction (result_type (IMPLICIT_DEF))),
              (instruction op_reg_class:$rs1),
              (instruction op_reg_class:$rs2),
              (instruction (mask_type zero_reg),
               VLMax,
               sew),
              swap>.Value>;
}

multiclass pat_vop_binary_v_nomask<SDNode vop,
                                   string instruction_name,
                                   string kind,
                                   ValueType result_type,
                                   ValueType op_type,
                                   int sew,
                                   int vlmul,
                                   VReg op_reg_class,
                                   bit swap = 0>
{
  foreach instruction = [!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)] in
  def : Pat<(result_type (vop
                          (op_type op_reg_class:$rs1),
                          (op_type op_reg_class:$rs2))),
            swap_helper<
              (instruction), // empty prefix
              (instruction op_reg_class:$rs1),
              (instruction op_reg_class:$rs2),
              (instruction
               VLMax,
               sew),
              swap>.Value>;
}

multiclass pat_vop_binary_xif<SDNode vop,
                              string instruction_name,
                              string kind,
                              ValueType result_type,
                              ValueType op1_type,
                              ValueType op2_type,
                              ValueType mask_type,
                              int sew,
                              int vlmul,
                              VReg op_reg_class,
                              DAGOperand op2_kind,
                              bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name#_#kind#"_M"#vlmul);
  def : Pat<(result_type (vop
                          (op1_type op_reg_class:$rs1),
                          (op1_type (splat_vector op2_kind:$rs2)))),
            swap_helper<
              (instruction (result_type (IMPLICIT_DEF))),
              (instruction op_reg_class:$rs1),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction (mask_type zero_reg),
               VLMax,
               sew),
              swap>.Value>;
}

multiclass pat_vop_binary_int_v_vv_vx<SDNode vop,
                                      string instruction_name>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Vector, vti.Vector, vti.Mask,
                            vti.SEW, vti.RegClass.VLMul,
                            vti.RegClass>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VX",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.RegClass.VLMul,
                              vti.RegClass, GPR>;
  }
}

multiclass pat_vop_binary_int_v_vv_vx_vi<SDNode vop,
                                         string instruction_name,
                                         DAGOperand imm_kind = simm5>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Vector, vti.Vector, vti.Mask,
                            vti.SEW, vti.RegClass.VLMul,
                            vti.RegClass>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VX",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.RegClass.VLMul,
                              vti.RegClass, GPR>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VI",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.RegClass.VLMul,
                              vti.RegClass, imm_kind>;
  }
}

multiclass pat_vop_binary_fp_v_vv_vf<SDNode vop,
                                     string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            fvti.Vector, fvti.Vector, fvti.Mask,
                            fvti.SEW, fvti.RegClass.VLMul,
                            fvti.RegClass>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VF",
                              fvti.Vector, fvti.Vector, fvti.Scalar, fvti.Mask,
                              fvti.SEW, fvti.RegClass.VLMul,
                              fvti.RegClass, fvti.ScalarRegClass>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_binary_reduction_int<SDNode vop,
                                        string instruction_name>
{
  foreach vti = NoGroupIntegerVectors in {
    defvar instruction =
      !cast<Instruction>(instruction_name#"_VS_M1");
    def : Pat<(vop (vti.Vector vti.RegClass:$rs2)),
              (PseudoVMV_X_S
               (vti.Vector (instruction
                (vti.Vector (IMPLICIT_DEF)),
                (vti.Vector vti.RegClass:$rs2),
                (PseudoVMV_S_X
                 (vti.Vector (IMPLICIT_DEF)),
                 X0,
                 VLMax, vti.SEW),
                (vti.Mask zero_reg),
                VLMax, vti.SEW)),
               vti.SEW)>;
  }
  foreach gvti = GroupIntegerVectors in {
    defvar instruction =
      !cast<Instruction>(instruction_name#"_VS_M"#gvti.RegClass.VLMul);
    def : Pat<(vop (gvti.Vector gvti.RegClass:$rs2)),
              (PseudoVMV_X_S
               (EXTRACT_SUBREG
                (gvti.Vector (instruction
                 (gvti.Vector (IMPLICIT_DEF)),
                 (gvti.Vector gvti.RegClass:$rs2),
                 (INSERT_SUBREG
                  (IMPLICIT_DEF),
                  (PseudoVMV_S_X
                   (gvti.VectorM1 (IMPLICIT_DEF)),
                   X0,
                   VLMax, gvti.SEW),
                  sub_vrm2),
                 (gvti.Mask zero_reg),
                 VLMax, gvti.SEW)),
                sub_vrm2),
               gvti.SEW)>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_binary_reduction_fp<SDNode vop,
                                       string instruction_name>
{
  foreach fvti = NoGroupFloatVectors in {
    defvar instruction =
      !cast<Instruction>(instruction_name#"_VS_M1");
    def : Pat<(vop (fvti.Vector fvti.RegClass:$rs2)),
              FromFPR64<fvti.ScalarRegClass,
               (PseudoVFMV_F_S
                (fvti.Vector (instruction
                 (fvti.Vector (IMPLICIT_DEF)),
                 (fvti.Vector fvti.RegClass:$rs2),
                 (PseudoVFMV_S_F
                  (fvti.Vector (IMPLICIT_DEF)),
                  ToFPR64<fvti.ScalarRegClass, FPZero<fvti.ScalarRegClass>.ret>.ret,
                  VLMax, fvti.SEW),
                 (fvti.Mask zero_reg),
                 VLMax, fvti.SEW)),
                fvti.SEW)>.ret>;
  }
  foreach gfvti = GroupFloatVectors in {
    defvar instruction =
      !cast<Instruction>(instruction_name#"_VS_M"#gfvti.RegClass.VLMul);
    def : Pat<(vop (gfvti.Vector gfvti.RegClass:$rs2)),
              FromFPR64<gfvti.ScalarRegClass,
               (PseudoVFMV_F_S
                (EXTRACT_SUBREG
                 (gfvti.Vector (instruction
                  (gfvti.Vector (IMPLICIT_DEF)),
                  (gfvti.Vector gfvti.RegClass:$rs2),
                  (INSERT_SUBREG
                   (IMPLICIT_DEF),
                   (PseudoVFMV_S_F
                    (gfvti.VectorM1 (IMPLICIT_DEF)),
                    ToFPR64<gfvti.ScalarRegClass, FPZero<gfvti.ScalarRegClass>.ret>.ret,
                    VLMax, gfvti.SEW),
                   sub_vrm2),
                  (gfvti.Mask zero_reg),
                  VLMax, gfvti.SEW)),
                 sub_vrm2),
                gfvti.SEW)>.ret>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_binary_reduction_v2_fp<SDNode vop,
                                          string instruction_name>
{
  foreach fvti = NoGroupFloatVectors in {
    defvar instruction =
      !cast<Instruction>(instruction_name#"_VS_M1");
    def : Pat<(vop (fvti.Scalar fvti.ScalarRegClass:$init),
                   (fvti.Vector fvti.RegClass:$rs2)),
              FromFPR64<fvti.ScalarRegClass,
               (PseudoVFMV_F_S
                (fvti.Vector (instruction
                 (fvti.Vector (IMPLICIT_DEF)),
                 (fvti.Vector fvti.RegClass:$rs2),
                 (PseudoVFMV_S_F
                  (fvti.Vector (IMPLICIT_DEF)),
                  ToFPR64<fvti.ScalarRegClass,
                          (fvti.Scalar fvti.ScalarRegClass:$init)>.ret,
                  VLMax, fvti.SEW),
                 (fvti.Mask zero_reg),
                 VLMax, fvti.SEW)),
                fvti.SEW)>.ret>;
  }
  foreach gfvti = GroupFloatVectors in {
    defvar instruction =
      !cast<Instruction>(instruction_name#"_VS_M"#gfvti.RegClass.VLMul);
    def : Pat<(vop (gfvti.Scalar gfvti.ScalarRegClass:$init),
                   (gfvti.Vector gfvti.RegClass:$rs2)),
              FromFPR64<gfvti.ScalarRegClass,
               (PseudoVFMV_F_S
                (EXTRACT_SUBREG
                 (gfvti.Vector (instruction
                  (gfvti.Vector (IMPLICIT_DEF)),
                  (gfvti.Vector gfvti.RegClass:$rs2),
                  (INSERT_SUBREG
                   (IMPLICIT_DEF),
                   (PseudoVFMV_S_F
                    (gfvti.VectorM1 (IMPLICIT_DEF)),
                    ToFPR64<gfvti.ScalarRegClass,
                            (gfvti.Scalar gfvti.ScalarRegClass:$init)>.ret,
                    VLMax, gfvti.SEW),
                   sub_vrm2),
                  (gfvti.Mask zero_reg),
                  VLMax, gfvti.SEW)),
                 sub_vrm2),
                gfvti.SEW)>.ret>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_ternary_v<SDNode vop,
                             string instruction_name,
                             ValueType vector_type,
                             ValueType mask_type,
                             int sew,
                             int vlmul,
                             VReg vector_reg_class>
{
  def : Pat<(vector_type (vop
                          (vector_type vector_reg_class:$rs3),
                          (vector_type vector_reg_class:$rs1),
                          (vector_type vector_reg_class:$rs2))),
            (!cast<Instruction>(instruction_name#"_VV_M"#vlmul)
             vector_reg_class:$rs3,
             vector_reg_class:$rs1,
             vector_reg_class:$rs2,
             (mask_type zero_reg),
             VLMax,
             sew)>;
}

multiclass pat_vop_ternary_xf<SDNode vop,
                              string instruction_name,
                              string kind,
                              ValueType vector_type,
                              ValueType op1_type,
                              ValueType mask_type,
                              int sew,
                              int vlmul,
                              VReg vector_reg_class,
                              DAGOperand op1_kind>
{
  def : Pat<(vector_type (vop
                          (vector_type vector_reg_class:$rs3),
                          (vector_type (splat_vector op1_kind:$rs1)),
                          (vector_type vector_reg_class:$rs2))),
            (!cast<Instruction>(instruction_name#_#kind#"_M"#vlmul)
             vector_reg_class:$rs3,
             ToFPR64<op1_kind, (op1_type op1_kind:$rs1)>.ret,
             vector_reg_class:$rs2,
             (mask_type zero_reg),
             VLMax,
             sew)>;
}

multiclass pat_vop_ternary_fp_v_vv_vf<SDNode vop,
                                      string instruction_name>
{
  foreach fvti = AllFloatVectors in
  {
    defm : pat_vop_ternary_v<vop, instruction_name,
                             fvti.Vector, fvti.Mask,
                             fvti.SEW, fvti.RegClass.VLMul,
                             fvti.RegClass>;
    defm : pat_vop_ternary_xf<vop, instruction_name, "VF",
                              fvti.Vector, fvti.Scalar, fvti.Mask,
                              fvti.SEW, fvti.RegClass.VLMul,
                              fvti.RegClass, fvti.ScalarRegClass>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_unary<SDNode vop,
                         string instruction_name,
                         ValueType vector_type,
                         ValueType mask_type,
                         int sew,
                         int vlmul,
                         VReg vector_reg_class>
{
  def : Pat<(vector_type (vop
                          (vector_type vector_reg_class:$rs2))),
            (!cast<Instruction>(instruction_name#"_V_M"#vlmul)
             (vector_type (IMPLICIT_DEF)),
             vector_reg_class:$rs2,
             (mask_type zero_reg),
             VLMax,
             sew)>;
}

multiclass pat_vop_unary_fp_v_v<SDNode vop,
                                string instruction_name>
{
  foreach vti = AllFloatVectors in
  {
    defm : pat_vop_unary<vop, instruction_name,
                         vti.Vector, vti.Mask,
                         vti.SEW, vti.RegClass.VLMul,
                         vti.RegClass>;
  }
}

let Predicates = [HasStdExtV] in {

defm "" : pat_vop_binary_int_v_vv_vx_vi<add, "PseudoVADD">;
defm "" : pat_vop_binary_int_v_vv_vx_vi<shl, "PseudoVSLL", uimm5>;
defm "" : pat_vop_binary_int_v_vv_vx_vi<and, "PseudoVAND">;
defm "" : pat_vop_binary_int_v_vv_vx<sub, "PseudoVSUB">;
defm "" : pat_vop_binary_int_v_vv_vx<mul, "PseudoVMUL">;

defm "" : pat_vop_binary_fp_v_vv_vf<fadd, "PseudoVFADD">;
defm "" : pat_vop_binary_fp_v_vv_vf<fsub, "PseudoVFSUB">;
defm "" : pat_vop_binary_fp_v_vv_vf<fmul, "PseudoVFMUL">;
// FIXME: Our understanding from the spec is that the following is wrong and
// should be 'fminimum' and 'fmaximum' instead. However this issue is found in
// the base operations in F and D so for now we keep it coherent with those.
defm "" : pat_vop_binary_fp_v_vv_vf<fminnum, "PseudoVFMIN">;
defm "" : pat_vop_binary_fp_v_vv_vf<fmaxnum, "PseudoVFMAX">;

defm "" : pat_vop_binary_reduction_int<vecreduce_add, "PseudoVREDSUM">;
defm "" : pat_vop_binary_reduction_int<vecreduce_and, "PseudoVREDAND">;
defm "" : pat_vop_binary_reduction_int<vecreduce_or, "PseudoVREDOR">;
defm "" : pat_vop_binary_reduction_int<vecreduce_xor, "PseudoVREDXOR">;
defm "" : pat_vop_binary_reduction_int<vecreduce_smax, "PseudoVREDMAX">;
defm "" : pat_vop_binary_reduction_int<vecreduce_smin, "PseudoVREDMIN">;
defm "" : pat_vop_binary_reduction_int<vecreduce_umax, "PseudoVREDMAXU">;
defm "" : pat_vop_binary_reduction_int<vecreduce_umin, "PseudoVREDMINU">;

defm "" : pat_vop_binary_reduction_v2_fp<vecreduce_strict_fadd, "PseudoVFREDOSUM">;
defm "" : pat_vop_binary_reduction_fp<vecreduce_fadd, "PseudoVFREDSUM">;
defm "" : pat_vop_binary_reduction_fp<vecreduce_fmax, "PseudoVFREDMAX">;
defm "" : pat_vop_binary_reduction_fp<vecreduce_fmin, "PseudoVFREDMIN">;

defm "" : pat_vop_ternary_fp_v_vv_vf<fma, "PseudoVFMADD">;

defm "" : pat_vop_unary_fp_v_v<fsqrt, "PseudoVFSQRT">;

foreach fvti = AllFloatVectors in
{
  defvar ivti = GetIntVectorTypeInfo<fvti>.Vti;

  def : Pat<(fvti.Vector (splat_vector fvti.ScalarRegClass:$value)),
            (!cast<Instruction>("PseudoVFMV_V_F_M"#fvti.RegClass.VLMul)
             ToFPR64<fvti.ScalarRegClass,
                     (fvti.Scalar fvti.ScalarRegClass:$value)>.ret,
             VLMax,
             fvti.SEW)>;

  def : Pat<(fvti.Vector (sint_to_fp (ivti.Vector ivti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_F_X_V_M" # fvti.RegClass.VLMul)
             (fvti.Vector (IMPLICIT_DEF)),
             $value,
             (ivti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  def : Pat<(fvti.Vector (uint_to_fp (ivti.Vector ivti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_F_XU_V_M" # fvti.RegClass.VLMul)
             (fvti.Vector (IMPLICIT_DEF)),
             $value,
             (ivti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  def : Pat<(ivti.Vector (fp_to_sint (fvti.Vector fvti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_X_F_V_M" # fvti.RegClass.VLMul)
             (ivti.Vector (IMPLICIT_DEF)),
             $value,
             (fvti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  def : Pat<(ivti.Vector (fp_to_uint (fvti.Vector fvti.RegClass:$value))),
            (!cast<Instruction>("PseudoVFCVT_XU_F_V_M" # fvti.RegClass.VLMul)
             (ivti.Vector (IMPLICIT_DEF)),
             $value,
             (fvti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;
}

foreach vti = AllIntegerVectors in
{
  def : Pat<(vti.Vector (splat_vector GPR:$value)),
            (!cast<Instruction>("PseudoVMV_V_X_M"#vti.RegClass.VLMul)
             GPR:$value,
             VLMax, vti.SEW)>;
  def : Pat<(vti.Vector (splat_vector simm5:$value)),
            (!cast<Instruction>("PseudoVMV_V_I_M"#vti.RegClass.VLMul)
             simm5:$value,
             VLMax, vti.SEW)>;

  def : Pat<(vti.Vector (int_experimental_vector_stepvector)),
            (!cast<Instruction>("PseudoVID_V_M"#vti.RegClass.VLMul)
             (vti.Vector (IMPLICIT_DEF)),
             (vti.Mask zero_reg),
             VLMax, vti.SEW)>;
}

foreach mti = AllMasks in {
  def : Pat<(mti.Mask (splat_vector (i64 1))),
            (!cast<Instruction>("PseudoVMSET_M"#mti.VLMul) VLMax, mti.SEW)>;
  def : Pat<(mti.Mask (splat_vector (i64 0))),
            (!cast<Instruction>("PseudoVMCLR_M"#mti.VLMul) VLMax, mti.SEW)>;
}

foreach vti = NoGroupIntegerVectors in {
  // Note: It is not valid in extractelt to have a return type that is different
  // to the vector element type. For those cases (EltTy != i64) we use a custom
  // node. There is another (deprecated) node vector_extract that won't fail in
  // tablegen, but it won't emit a matcher, so it is not usable either.
  def : Pat<(i64 (!cast<SDNode>(!if(!eq(vti.SEW, 64),
                                 extractelt, riscv_extract_vector_elt))
                   (vti.Vector vti.RegClass:$rs2),
                   GPR:$rs1)),
            (PseudoVMV_X_S
             (PseudoVSLIDEDOWN_VX_M1
              (vti.Vector (IMPLICIT_DEF)),
              (vti.Vector vti.RegClass:$rs2),
              GPR:$rs1,
              (vti.Mask zero_reg),
              VLMax,
              vti.SEW),
             vti.SEW)>;

  def : Pat<(i64 (!cast<SDNode>(!if(!eq(vti.SEW, 64),
                                 extractelt, riscv_extract_vector_elt))
                  (vti.Vector vti.RegClass:$rs2), 0)),
            (PseudoVMV_X_S
             (vti.Vector vti.RegClass:$rs2),
             vti.SEW)>;

  // Note: We use the (deprecated) node vector_insert when the input element
  // type is different to the result vector element type, since this is invalid
  // for insertelt.
  def : Pat<(vti.Vector (!cast<SDNode>(!if(!eq(vti.SEW, 64),
                                        insertelt, vector_insert))
                         (vti.Vector vti.RegClass:$merge),
                         GPR:$rs1, 0)),
            (PseudoVMV_S_X
             (vti.Vector vti.RegClass:$merge),
             GPR:$rs1,
             VLMax,
             vti.SEW)>;
}

foreach fvti = NoGroupFloatVectors in {
  def : Pat<(fvti.Scalar (extractelt (fvti.Vector fvti.RegClass:$rs2),
                          GPR:$rs1)),
            FromFPR64<fvti.ScalarRegClass,
                      (PseudoVFMV_F_S
                       (PseudoVSLIDEDOWN_VX_M1
                        (fvti.Vector (IMPLICIT_DEF)),
                        (fvti.Vector fvti.RegClass:$rs2),
                        GPR:$rs1,
                        (fvti.Mask zero_reg),
                        VLMax,
                        fvti.SEW),
                       fvti.SEW)>.ret>;

  def : Pat<(fvti.Scalar (extractelt (fvti.Vector fvti.RegClass:$rs2), 0)),
            FromFPR64<fvti.ScalarRegClass,
                      (PseudoVFMV_F_S $rs2, fvti.SEW)>.ret>;

  def : Pat<(fvti.Vector (insertelt (fvti.Vector fvti.RegClass:$merge),
                         (fvti.Scalar fvti.ScalarRegClass:$rs1), 0)),
            (PseudoVFMV_S_F
             (fvti.Vector fvti.RegClass:$merge),
             ToFPR64<fvti.ScalarRegClass,
                     (fvti.Scalar fvti.ScalarRegClass:$rs1)>.ret,
             VLMax, fvti.SEW)>;
}

foreach gvti = GroupIntegerVectors in {
  // Note: It is not valid in extractelt to have a return type that is different
  // to the vector element type. For those cases (EltTy != i64) we use a custom
  // node. There is another (deprecated) node vector_extract that won't fail in
  // tablegen, but it won't emit a matcher, so it is not usable either.
  def : Pat<(i64 (!cast<SDNode>(!if(!eq(gvti.SEW, 64),
                                 extractelt,
                                 riscv_extract_vector_elt))
                   (gvti.Vector gvti.RegClass:$rs2),
                   GPR:$rs1)),
            (PseudoVMV_X_S
             (EXTRACT_SUBREG
              (!cast<Instruction>("PseudoVSLIDEDOWN_VX_M"#gvti.RegClass.VLMul)
               (gvti.Vector (IMPLICIT_DEF)),
               (gvti.Vector gvti.RegClass:$rs2),
               GPR:$rs1,
               (gvti.Mask zero_reg),
               VLMax,
               gvti.SEW),
              sub_vrm2),
             gvti.SEW)>;

  def : Pat<(i64 (!cast<SDNode>(!if(!eq(gvti.SEW, 64),
                                 extractelt,
                                 riscv_extract_vector_elt))
                   (gvti.Vector gvti.RegClass:$rs2), 0)),
            (PseudoVMV_X_S
             (EXTRACT_SUBREG
              (gvti.Vector gvti.RegClass:$rs2),
              sub_vrm2),
             gvti.SEW)>;

  // Note: We use the (deprecated) node vector_insert when the input element
  // type is different to the result vector element type, since this is invalid
  // for insertelt.
  def : Pat<(gvti.Vector (!cast<SDNode>(!if(!eq(gvti.SEW, 64),
                                         insertelt, vector_insert))
                         (gvti.Vector gvti.RegClass:$merge),
                         GPR:$rs1, 0)),
            (INSERT_SUBREG (IMPLICIT_DEF),
             (PseudoVMV_S_X
              (EXTRACT_SUBREG
               (gvti.Vector gvti.RegClass:$merge),
               sub_vrm2),
              GPR:$rs1,
              VLMax,
              gvti.SEW),
             sub_vrm2)>;
}

foreach gfvti = GroupFloatVectors in {
  def : Pat<(gfvti.Scalar (extractelt (gfvti.Vector gfvti.RegClass:$rs2),
                           GPR:$rs1)),
            FromFPR64<gfvti.ScalarRegClass,
                      (PseudoVFMV_F_S
                       (EXTRACT_SUBREG
                        (!cast<Instruction>("PseudoVSLIDEDOWN_VX_M"#gfvti.RegClass.VLMul)
                         (gfvti.Vector (IMPLICIT_DEF)),
                         (gfvti.Vector gfvti.RegClass:$rs2),
                         GPR:$rs1,
                         (gfvti.Mask zero_reg),
                         VLMax,
                         gfvti.SEW),
                        sub_vrm2),
                       gfvti.SEW)>.ret>;

  def : Pat<(gfvti.Scalar (extractelt (gfvti.Vector gfvti.RegClass:$rs2), 0)),
            FromFPR64<gfvti.ScalarRegClass,
                      (PseudoVFMV_F_S
                       (EXTRACT_SUBREG
                        (gfvti.Vector gfvti.RegClass:$rs2),
                        sub_vrm2),
                       gfvti.SEW)>.ret>;

  def : Pat<(gfvti.Vector (insertelt (gfvti.Vector gfvti.RegClass:$merge),
                          (gfvti.Scalar gfvti.ScalarRegClass:$rs1), 0)),
            (INSERT_SUBREG (IMPLICIT_DEF),
             (PseudoVFMV_S_F
              (EXTRACT_SUBREG
               (gfvti.Vector gfvti.RegClass:$merge),
               sub_vrm2),
              ToFPR64<gfvti.ScalarRegClass,
                      (gfvti.Scalar gfvti.ScalarRegClass:$rs1)>.ret,
              VLMax,
              gfvti.SEW),
             sub_vrm2)>;
}

foreach vti = AllIllegalIntegerVectors in {
  defvar lvti = vti.LegalVti;

  def : Pat<(lvti.Vector (riscv_sign_extend_inreg (lvti.Vector VR:$rs2),
                          uimm5:$rs1)),
            (PseudoVSRA_VI_M1
             (lvti.Vector (IMPLICIT_DEF)),
             (PseudoVSLL_VI_M1
              (lvti.Vector (IMPLICIT_DEF)),
              $rs2, uimm5:$rs1,
              (lvti.Mask zero_reg),
              VLMax,
              lvti.SEW),
             uimm5:$rs1,
             (lvti.Mask zero_reg),
             VLMax,
             lvti.SEW)>;
}

foreach vti = AllIllegalIntegerVectors in {
  defvar lvti = vti.LegalVti;

  def : Pat<(lvti.Vector (riscv_sign_extend_inreg (lvti.Vector VR:$rs2),
                          GPR:$rs1)),
            (PseudoVSRA_VX_M1
             (lvti.Vector (IMPLICIT_DEF)),
             (PseudoVSLL_VX_M1
              (lvti.Vector (IMPLICIT_DEF)),
              $rs2, $rs1,
              (lvti.Mask zero_reg),
              VLMax,
              lvti.SEW),
             $rs1,
             (lvti.Mask zero_reg),
             VLMax,
             lvti.SEW)>;
}

foreach vti = AllVectors in {
  def : Pat<(vti.Vector (vselect (vti.Mask V0), (vti.Vector vti.RegClass:$rs2), (vti.Vector vti.RegClass:$rs1))),
            (!cast<Instruction>("PseudoVMERGE_VVM_M"#vti.RegClass.VLMul)
             (vti.Vector vti.RegClass:$rs1),
             (vti.Vector vti.RegClass:$rs2),
             (vti.Mask V0),
             VLMax, vti.SEW)>;
}

foreach fvti = AllFloatVectors in {
  def : Pat<(fabs (fvti.Vector fvti.RegClass:$rs2)),
            (!cast<Instruction>("PseudoVFSGNJX_VV_M"#fvti.RegClass.VLMul)
             (fvti.Vector (IMPLICIT_DEF)),
             (fvti.Vector fvti.RegClass:$rs2),
             (fvti.Vector fvti.RegClass:$rs2),
             (fvti.Mask zero_reg),
             VLMax, fvti.SEW)>;
}

}

//===----------------------------------------------------------------------===//
// Patterns. Full vector relational operations
//===----------------------------------------------------------------------===//

multiclass pat_vop_binary_int_m_vv<SDNode vop,
                                   string instruction_name,
                                   bit swap = 0>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Mask, vti.Vector, vti.Mask,
                            vti.SEW, vti.RegClass.VLMul,
                            vti.RegClass, swap>;
  }
}

multiclass pat_vop_binary_int_m_vv_vx<SDNode vop,
                                      string instruction_name,
                                      bit swap = 0>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Mask, vti.Vector, vti.Mask,
                            vti.SEW, vti.RegClass.VLMul,
                            vti.RegClass, swap>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VX",
                              vti.Mask, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.RegClass.VLMul,
                              vti.RegClass, GPR, swap>;
  }
}

multiclass pat_vop_binary_int_m_vv_vx_vi<SDNode vop,
                                         string instruction_name,
                                         bit swap = 0>
{
  foreach vti = AllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Mask, vti.Vector, vti.Mask,
                            vti.SEW, vti.RegClass.VLMul,
                            vti.RegClass, swap>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VX",
                              vti.Mask, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.RegClass.VLMul,
                              vti.RegClass, GPR, swap>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VI",
                              vti.Mask, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.RegClass.VLMul,
                              vti.RegClass, simm5, swap>;
  }
}

let Predicates = [HasStdExtV] in {

defm "" : pat_vop_binary_int_m_vv_vx_vi<seteq, "PseudoVMSEQ">;
defm "" : pat_vop_binary_int_m_vv_vx_vi<setne, "PseudoVMSNE">;
defm "" : pat_vop_binary_int_m_vv_vx_vi<setule, "PseudoVMSLEU">;
defm "" : pat_vop_binary_int_m_vv_vx_vi<setle, "PseudoVMSLE">;
defm "" : pat_vop_binary_int_m_vv_vx<setult, "PseudoVMSLTU">;
defm "" : pat_vop_binary_int_m_vv_vx<setlt, "PseudoVMSLT">;
defm "" : pat_vop_binary_int_m_vv<setugt, "PseudoVMSLTU", /* swap */ 1>;
defm "" : pat_vop_binary_int_m_vv<setgt, "PseudoVMSLT", /* swap */ 1>;
defm "" : pat_vop_binary_int_m_vv<setuge, "PseudoVMSLEU", /* swap */ 1>;
defm "" : pat_vop_binary_int_m_vv<setge, "PseudoVMSLE", /* swap */ 1>;

}

//===----------------------------------------------------------------------===//
// Patterns. Full vector conversions
//===----------------------------------------------------------------------===//

foreach vtiToWti = AllWideableIntVectors in
{
  defvar vti = vtiToWti.Vti;
  defvar wti = vtiToWti.Wti;

  // sext (SEW->2*SEW)
  def : Pat<(wti.Vector (riscv_sign_extend_vector (vti.Vector vti.RegClass:$rs1))),
             (!cast<Instruction>("PseudoVWADD_VX_M" # vti.RegClass.VLMul)
               (wti.Vector (IMPLICIT_DEF)),
               (vti.Vector $rs1),
               X0,
               (vti.Mask zero_reg),
               X0,
               vti.SEW)>;

  // zext (SEW->2*SEW)
  def : Pat<(wti.Vector (riscv_zero_extend_vector (vti.Vector vti.RegClass:$rs1))),
             (!cast<Instruction>("PseudoVWADDU_VX_M" # vti.RegClass.VLMul)
               (wti.Vector (IMPLICIT_DEF)),
               (vti.Vector $rs1),
               X0,
               (vti.Mask zero_reg),
               X0,
               vti.SEW)>;

  // trunc (2*SEW->SEW)
  def : Pat<(vti.Vector (riscv_trunc_vector (wti.Vector wti.RegClass:$rs1))),
             (!cast<Instruction>("PseudoVNSRL_WI_M" # vti.RegClass.VLMul)
               (vti.Vector (IMPLICIT_DEF)),
               (wti.Vector $rs1),
               0,
               (vti.Mask zero_reg),
               X0,
               vti.SEW)>;

  // FIXME: Missing remaining conversions.
}

//===----------------------------------------------------------------------===//
// Patterns. Full vector mask logical operations
//===----------------------------------------------------------------------===//

multiclass pat_vop_binary_m_mm<SDNode vop,
                               string instruction_name>
{
  foreach mti = AllMasks in
  {
    defm : pat_vop_binary_v_nomask<vop, instruction_name, "MM",
                                   mti.Mask, mti.Mask,
                                   mti.SEW, mti.VLMul,
                                   VR>;
  }
}

let Predicates = [HasStdExtV] in {

defm "" : pat_vop_binary_m_mm<and, "PseudoVMAND">;
defm "" : pat_vop_binary_m_mm<xor, "PseudoVMXOR">;
defm "" : pat_vop_binary_m_mm<or, "PseudoVMOR">;

}

//===----------------------------------------------------------------------===//
// Patterns. Load/store
//===----------------------------------------------------------------------===//

multiclass pat_load {
  foreach vti = AllVectors in {
    defvar ivti = GetIntVectorTypeInfo<vti>.Vti;
    defvar vlmul = vti.RegClass.VLMul;

    def : Pat<(vti.Vector (int_epi_vload GPR:$rs1, GPR:$vl)),
              (!cast<Instruction>("PseudoVLE"#vti.SEW#"_V_M"#vlmul)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_strided GPR:$rs1, GPR:$rs2, GPR:$vl)),
              (!cast<Instruction>("PseudoVLSE"#vti.SEW#"_V_M"#vlmul)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1, $rs2,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_indexed GPR:$rs1,
                           (ivti.Vector vti.RegClass:$rs2), GPR:$vl)),
              (!cast<Instruction>("PseudoVLXEI"#vti.SEW#"_V_M"#vlmul)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1, $rs2,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_mask (vti.Vector vti.RegClass:$merge),
                           GPR:$rs1, (vti.Mask V0), GPR:$vl)),
               (!cast<Instruction>("PseudoVLE"#vti.SEW#"_V_M"#vlmul)
                $merge,
                $rs1,
                (vti.Mask V0),
                (NoX0 GPR:$vl),
                vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_strided_mask
                           (vti.Vector vti.RegClass:$merge), GPR:$rs1, GPR:$rs2,
                           (vti.Mask V0), GPR:$vl)),
              (!cast<Instruction>("PseudoVLSE"#vti.SEW#"_V_M"#vlmul)
               $merge,
               $rs1, $rs2,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_indexed_mask
                           (vti.Vector vti.RegClass:$merge), GPR:$rs1,
                           (ivti.Vector vti.RegClass:$rs2), (vti.Mask V0),
                           GPR:$vl)),
              (!cast<Instruction>("PseudoVLXEI"#vti.SEW#"_V_M"#vlmul)
               $merge,
               $rs1, $rs2,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;
  }
}

multiclass pat_store {
  foreach vti = AllVectors in {
    defvar ivti = GetIntVectorTypeInfo<vti>.Vti;
    defvar vlmul = vti.RegClass.VLMul;

    def : Pat<(int_epi_vstore (vti.Vector vti.RegClass:$rs3), GPR:$rs1,
               GPR:$vl),
              (!cast<Instruction>("PseudoVSE"#vti.SEW#"_V_M"#vlmul)
               $rs3, $rs1,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_strided
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, GPR:$rs2, GPR:$vl),
              (!cast<Instruction>("PseudoVSSE"#vti.SEW#"_V_M"#vlmul)
               $rs3, $rs1, $rs2,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_indexed
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1,
               (ivti.Vector vti.RegClass:$rs2), GPR:$vl),
              (!cast<Instruction>("PseudoVSXEI"#vti.SEW#"_V_M"#vlmul)
               $rs3, $rs1, $rs2,
               (vti.Mask zero_reg),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, (vti.Mask V0),
               GPR:$vl),
              (!cast<Instruction>("PseudoVSE"#vti.SEW#"_V_M"#vlmul)
               $rs3, $rs1,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_strided_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1, GPR:$rs2,
               (vti.Mask V0), GPR:$vl),
              (!cast<Instruction>("PseudoVSSE"#vti.SEW#"_V_M"#vlmul)
               $rs3, $rs1, $rs2,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;

    def : Pat<(int_epi_vstore_indexed_mask
               (vti.Vector vti.RegClass:$rs3), GPR:$rs1,
               (ivti.Vector vti.RegClass:$rs2), (vti.Mask V0), GPR:$vl),
              (!cast<Instruction>("PseudoVSXEI"#vti.SEW#"_V_M"#vlmul)
               $rs3, $rs1, $rs2,
               (vti.Mask V0),
               (NoX0 GPR:$vl),
               vti.SEW)>;
  }
}

let Predicates = [HasStdExtV] in {

def : Pat<(nxv1f64 (masked_ld GPR:$addr, undef, (nxv1i1 V0), (nxv1f64 VR:$merge))),
          (PseudoVLE64_V_M1
             (nxv1f64 VR:$merge),
             GPR:$addr,
             (nxv1i1 V0),
             /* vl */ X0,
             64)>;

def : Pat<(masked_st (nxv1f64 VR:$value), GPR:$addr, undef, (nxv1i1 V0)),
          (PseudoVSE64_V_M1
              VR:$value, GPR:$addr, (nxv1i1 V0), /* vl */ X0, 64)>;

def : Pat<(nxv1i1 (setule (nxv1i64 VR:$rs1), (nxv1i64 VR:$rs2))),
          (PseudoVMSLEU_VV_M1 (nxv1i64 (IMPLICIT_DEF)), VR:$rs1, VR:$rs2,
           (nxv1i1 zero_reg), /* vl */ X0, 64)>;

defm "" : pat_load;
defm "" : pat_store;

}

let Predicates = [HasStdExtV] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, usesCustomInserter = 1 in
def PseudoImplicitVRTuple : Pseudo<(outs VRTuple:$rd), (ins), []>;

}

let Predicates = [HasStdExtV, HasStdExtZvlsseg] in {

foreach vti = NoGroupVectors in {
  def : Pat<(riscv_vlseg2 GPR:$rs1, GPR:$vl, (XLenVT vti.SEW)),
            (!cast<Instruction>("PseudoVLSEG2E"#vti.SEW#"_V_M1")
             (PseudoImplicitVRTuple),
             $rs1,
             (nxv1i1 zero_reg),
             (NoX0 GPR:$vl),
             vti.SEW)>;

  def : Pat<(riscv_vsseg2 VRTuple:$rs3, GPR:$rs1, GPR:$vl, (XLenVT vti.SEW)),
            (!cast<Instruction>("PseudoVSSEG2E"#vti.SEW#"_V_M1")
             $rs3, $rs1,
             (nxv1i1 zero_reg),
             (NoX0 GPR:$vl),
             vti.SEW)>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, usesCustomInserter = 1 in
def PseudoVBuildTuple2 : Pseudo<(outs VRTuple:$rd),
                                (ins VR:$rs1, VR:$rs2), []>;

}

//===----------------------------------------------------------------------===//
// EPI custom instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtV] in {

let VLMul = 1, hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    usesCustomInserter = 1 in {

defm PseudoVZIP2_VV : pseudo_binary<VRTuple, VR, VR, /* LMUL */ 1, "">;
defm PseudoVUNZIP2_VV : pseudo_binary<VRTuple, VR, VR, /* LMUL */ 1, "">;
defm PseudoVTRN_VV : pseudo_binary<VRTuple, VR, VR, /* LMUL */ 1, "">;

}

foreach vti = NoGroupVectors in {

    def : Pat<(riscv_vzip2 (vti.Vector VR:$rs1), (vti.Vector VR:$rs2),
                           (i64 GPR:$vl)),
              (PseudoVZIP2_VV_M1
               (PseudoImplicitVRTuple), vti.RegClass:$rs1, vti.RegClass:$rs2,
               (vti.Mask zero_reg), (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(riscv_vunzip2 (vti.Vector VR:$rs1), (vti.Vector VR:$rs2),
                             (i64 GPR:$vl)),
              (PseudoVUNZIP2_VV_M1
               (PseudoImplicitVRTuple), vti.RegClass:$rs1, vti.RegClass:$rs2,
               (vti.Mask zero_reg), (NoX0 GPR:$vl), vti.SEW)>;

    def : Pat<(riscv_vtrn (vti.Vector VR:$rs1), (vti.Vector VR:$rs2),
                          (i64 GPR:$vl)),
              (PseudoVTRN_VV_M1
               (PseudoImplicitVRTuple), vti.RegClass:$rs1, vti.RegClass:$rs2,
               (vti.Mask zero_reg), (NoX0 GPR:$vl), vti.SEW)>;
}

}
