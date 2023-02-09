#!/usr/bin/env python3
# coding=utf-8

import collections
import json
import logging
import string
import subprocess
import sys
import tempfile
import os

IMPLEMENTED_LMULS = [1, 2, 4, 8]

# Prototype kinds
PRIMARIES = "evwm0us"
MODIFIERS = "PKCUIFT"

runtime_error = False

# Types can be rendered in several ways
class Type:
    def __init__(self):
        pass

    def isPrimaryType(self):
        return False

    def isIntegerType(self):
        return False


class TypePrimary(Type):
    def __init__(self, letter):
        self.letter = letter

    def __str__(self):
        return self.letter

    def isPrimaryType(self):
        return True

    def isIntegerType(self):
        return self.letter in ["c", "s", "i", "l"]

    def render_for_name(self):
        if self.letter == "c":
            return "i8"
        elif self.letter == "s":
            return "i16"
        elif self.letter == "i":
            return "i32"
        elif self.letter == "l":
            return "i64"
        elif self.letter == "f":
            return "f32"
        elif self.letter == "d":
            return "f64"
        elif self.letter == "m":
            return "i1"
        else:
            raise Exception("Unknown letter {}".format(self.letter))

    def render_for_clang(self):
        if self.letter == "c":
            return "Sc"
        elif self.letter == "s":
            return "Ss"
        elif self.letter == "i":
            return "Si"
        elif self.letter == "l":
            return "SWi"
        elif self.letter == "f":
            return "f"
        elif self.letter == "d":
            return "d"
        elif self.letter == "m":
            return "b"
        else:
            raise Exception("Unknown letter {}".format(self.letter))


class TypeConstant(Type):
    def __init__(self, letter):
        self.letter = letter

    def __str__(self):
        return self.letter

    def render_for_name(self):
        raise Exception("No type constant can appear in the name")

    def render_for_clang(self):
        if self.letter == "0":
            return "v"
        elif self.letter == "s":
            return "SWi"
        elif self.letter == "u":
            return "UWi"
        else:
            raise Exception("Unknown type constant {}".format(self.letter))


class TypeVector(Type):
    def __init__(self, element_type, scale):
        self.element_type = element_type
        self.scale = scale

    def __str__(self):
        return "<" + str(self.scale) + " x " + str(self.element_type) + ">"

    def render_for_name(self):
        t = self.element_type.render_for_name()
        return "{}x{}".format(self.scale, t)

    def render_for_clang(self):
        t = self.element_type.render_for_clang()
        return "QV{}{}".format(self.scale, t)


class TypeTuple(Type):
    def __init__(self, tuple_size, vector_type):
        assert tuple_size > 0
        self.tuple_size = tuple_size
        self.vector_type = vector_type

    def __str__(self):
        return "{" + ", ".join([str(self.vector_type)] * self.tuple_size) + "}"

    def render_for_name(self):
        t = self.vector_type.render_for_name()
        return "{}x{}".format(t, self.tuple_size)

    def render_for_clang(self):
        t = self.vector_type.render_for_clang()
        return "{}{}".format("".join(["T"] * self.tuple_size), t)


class TypeModified(Type):
    def __init__(self, modified_type, modifier):
        self.modified_type = modified_type
        self.modifier = modifier

    def __str__(self):
        return self.modifier + str(self.modified_type)

    def isIntegerType(self):
        return self.modified_type.isIntegerType()

    def isPrimaryType(self):
        return self.modified_type.isPrimaryType()

    def render_for_name(self):
        raise Exception("No modified types appear in the name")

    def render_for_clang(self):
        t = self.modified_type.render_for_clang()
        if self.modifier == "P":
            return "{}*".format(t)
        elif self.modifier == "C":
            return "{}C".format(t)
        elif self.modifier == "K":
            return t.replace("i", "Ii")
        elif self.modifier == "U":
            return "U{}".format(t.replace("S", ""))
        else:
            raise Exception("Unexpected modified type {}".format(
                self.modifier))


# Prototypes are evaluated using a type spec and a LMUL to generate a Type.
class Prototype:
    def __init__(self):
        pass

    def evaluate(self, type_spec, lmul):
        raise Exception("Method 'evaluate' must be overriden")


class PrototypePrimary(Prototype):
    def __init__(self, letter):
        assert (letter in PRIMARIES)
        self.letter = letter

    def __str__(self):
        return self.letter

    def __computeVector(self, type_spec, lmul):
        if type_spec == "c":
            scale = 8
        elif type_spec == "s":
            scale = 4
        elif type_spec in ["f", "i"]:
            scale = 2
        elif type_spec in ["l", "d"]:
            scale = 1
        else:
            raise Exception("Unhandled type_spec '{}'".format(type_spec))
        scale *= lmul
        return TypeVector(TypePrimary(type_spec), scale)

    def __computeWideVector(self, type_spec, lmul):
        if type_spec == 'c':
            scale = 8
            base_type = TypePrimary("s")
        elif type_spec == "s":
            scale = 4
            base_type = TypePrimary("i")
        elif type_spec in ["f", "i"]:
            if type_spec == "f":
                base_type = TypePrimary("d")
            else:
                base_type = TypePrimary("l")
            scale = 2
        else:
            raise Exception("Unhandled type_spec '{}'".format(type_spec))
        scale *= lmul
        return TypeVector(base_type, scale)

    def __computeMaskVector(self, type_spec, lmul):
        if type_spec == 'c':
            scale = 8
        elif type_spec == "s":
            scale = 4
        elif type_spec in ["f", "i"]:
            scale = 2
        elif type_spec in ["l", "d"]:
            scale = 1
        else:
            raise Exception("Unhandled type_spec '{}'".format(type_spec))
        scale *= lmul
        return TypeVector(TypePrimary("m"), scale)

    def evaluate(self, type_spec, lmul):
        if self.letter == "e":
            return TypePrimary(type_spec)
        elif self.letter == "v":
            return self.__computeVector(type_spec, lmul)
        elif self.letter == "w":
            return self.__computeWideVector(type_spec, lmul)
        elif self.letter == "m":
            return self.__computeMaskVector(type_spec, lmul)
        elif self.letter in ["0", "u", "s"]:
            return TypeConstant(self.letter)
        else:
            raise Exception("Unhandled letter '{}'".format(self.letter))


class PrototypeModifier(Prototype):
    def __init__(self, letter, prototype):
        assert (letter in MODIFIERS)
        self.letter = letter
        self.prototype = prototype

    def __str__(self):
        return self.letter + str(self.prototype)

    def evaluate(self, type_spec, lmul):
        t = self.prototype.evaluate(type_spec, lmul)
        if self.letter == "P":
            if not t.isPrimaryType():
                raise Exception(
                    "P modifier can only be applied to primary types. Current type is '{}'"
                    .format(t))
            return TypeModified(t, self.letter)
        elif self.letter == "C":
            return TypeModified(t, self.letter)
        elif self.letter in ["U", "K"]:
            if not t.isIntegerType():
                raise Exception(
                    "U or K can only be applied to integer types. Current type is '{}'"
                    .format(t))
            return TypeModified(t, self.letter)
        elif self.letter in ["I", "F"]:
            if not isinstance(t, TypeVector):
                raise Exception(
                    "F or I can only be applied to vector types. Current type is '{}'"
                    .format(t))
            element_type = t.element_type
            if not isinstance(element_type, TypePrimary):
                raise Exception(
                    "Expecting a primary type in a vector but type is '{}".
                    format(element_type))
            if element_type.letter in ["f", "i"]:
                if self.letter == "F":
                    new_element_type = TypePrimary("f")
                else:
                    new_element_type = TypePrimary("i")
            elif element_type.letter in ["d", "l"]:
                if self.letter == "F":
                    new_element_type = TypePrimary("d")
                else:
                    new_element_type = TypePrimary("l")
            elif element_type.letter in ["c", "s"] and self.letter == "I":
                new_element_type = TypePrimary(element_type.letter)
            else:
                raise Exception(
                    "Cannot convert element type '{}' to floating or integer vector"
                    .format(element_type.letter))
            return TypeVector(new_element_type, t.scale)
        elif self.letter == "T":
            if not isinstance(t, TypeVector) and not isinstance(t, TypeTuple):
                raise Exception(
                    "T modifier can only be applied to vector types or tuples types. Current type is '{}'"
                    .format(t))
            tuple_size = 1
            vector_type = t
            if isinstance(t, TypeTuple):
                tuple_size = t.tuple_size + 1
                vector_type = t.vector_type
            return TypeTuple(tuple_size, vector_type)
        else:
            raise Exception("Unhandled modifier {}", self.letter)


def process_tblgen_file(tablegen, input_tablegen, include_paths):
    inc_paths = []
    for i in include_paths:
        inc_paths.append("-I")
        inc_paths.append(i)
    t = subprocess.check_output([tablegen, "--dump-json", input_tablegen] + inc_paths)
    return json.loads(t.decode("utf-8"))


def parse_single_prototype(prototype):
    # Reverse
    p = prototype[::-1]
    assert (p[0] in PRIMARIES)
    t = PrototypePrimary(p[0])
    p = p[1:]
    for l in p:
        assert (l in MODIFIERS)
        t = PrototypeModifier(l, t)
    return t


def parse_prototype_seq(prototype_seq):
    res = []
    current_prototype = ""
    for current_letter in prototype_seq:
        current_prototype += current_letter
        if current_letter in PRIMARIES:
            res.append(parse_single_prototype(current_prototype))
            current_prototype = ""
        elif current_letter in MODIFIERS:
            pass
        else:
            raise Exception("Invalid prototype letter {} in {}".format(
                current_letter, prototype_seq))

    if current_prototype != "":
        raise Exception("Prototype {} is incomplete".format(prototype_seq))
    return res


def render_type_for_name(prototype, type_spec, lmul):
    ty = prototype.evaluate(type_spec, lmul)
    return ty.render_for_name()


def render_types_for_name(suffix_types, type_spec, lmul):
    prototype = parse_prototype_seq(suffix_types)
    res = []
    for proto in prototype:
        res.append(render_type_for_name(proto, type_spec, lmul))
    return res


def render_type_for_clang(prototype, type_spec, lmul):
    ty = prototype.evaluate(type_spec, lmul)
    return ty.render_for_clang()


def compute_builtin_type_clang(builtin, prototype, type_spec, lmul):
    res = []
    for proto in prototype:
        res.append(render_type_for_clang(proto, type_spec, lmul))
    return "".join(res)


class InstantiatedBuiltin:
    def __init__(self, lmul, type_spec, full_name, type_description, flags, builtin,
            prototype, masked = False, index_of_mask = None):
        self.lmul = lmul
        self.type_spec = type_spec
        self.full_name = full_name
        self.type_description = type_description
        self.flags = flags
        self.builtin = builtin
        self.prototype = prototype
        self.masked = masked
        self.index_of_mask = index_of_mask

    def __str__(self):
        return "EPI_BUILTIN({}, \"{}\", \"{}\")".format(self.full_name, \
            self.type_description, self.flags)

    def c_prototoype_items(self):
        import builtin_parser
        import type_render
        (return_type, parameter_types) = builtin_parser.parse_type(self.type_description)
        return_type_str = type_render.TypeRender(return_type).render()
        parameter_types_str = map(lambda x : type_render.TypeRender(x).render(), parameter_types)

        return (return_type_str, parameter_types_str)

    def c_prototype(self, parameter_names):
        (return_type_str, parameter_types_str) = self.c_prototoype_items()

        # Now add names to the parameters
        # FIXME: This won't work for C declarators that need parentheses.
        # Luckily we don't need any yet.
        parameter_types_str = list(parameter_types_str) # python3 compatibility
        letter = ord('a')
        for i in range(len(parameter_types_str)):
            adjusted_i = i
            if self.masked and self.builtin["HasMergeOperand"]:
                adjusted_i -= 1

            if i == 0 and self.masked and self.builtin["HasMergeOperand"]:
                parameter_types_str[i] += " merge";
            elif self.index_of_mask is not None and \
                    self.index_of_mask - 1 == i and self.masked:
                parameter_types_str[i] += " mask";
            elif i + 1 == len(parameter_types_str) and self.builtin["HasVL"]:
                parameter_types_str[i] += " gvl";
            elif adjusted_i >= len(parameter_names):
                if letter == ord('z'):
                    raise Exception("Too many unnamed parameters")
                parameter_types_str[i] += " {}".format(chr(letter))
                letter += 1
            else:
                parameter_types_str[i] += " {}".format(parameter_names[adjusted_i])

        return "{} __builtin_epi_{}({});".format(return_type_str, \
            self.full_name, ", ".join(parameter_types_str))

def compute_builtin_name(builtin, prototype, type_spec, lmul):
    res = "{}".format(builtin["Name"])
    if not builtin["Suffix"]:
        return res
    rendered_types = render_types_for_name(builtin["Suffix"], type_spec, lmul)
    res += "_" + "_".join(rendered_types)
    return res


def compute_single_builtin_defs(builtin, orig_prototype, type_spec, lmul):
    builtin_list = []
    prototype = orig_prototype[:]

    if builtin["HasVL"] != 0:
        # Add GVL operand
        prototype.append(PrototypePrimary("u"))

    full_name = compute_builtin_name(builtin, prototype, type_spec, lmul)
    type_description = compute_builtin_type_clang(builtin, prototype,
                                                  type_spec, lmul)
    flags = ""
    if builtin["HasSideEffects"] == 0:
        flags = "n"

    builtin_list.append(InstantiatedBuiltin(lmul, type_spec, full_name, type_description,
        flags, builtin, prototype))

    if builtin["HasMask"] != 0:
        # Emit masked variant
        prototype_mask = orig_prototype[:]

        if builtin["HasMergeOperand"]:
            # Add merge operand
            prototype_mask.insert(1, prototype_mask[0])

        index_of_mask = len(prototype_mask)
        prototype_mask.append(PrototypePrimary("m"))

        if builtin["HasVL"] != 0:
            # Add GVL operand
            prototype_mask.append(PrototypePrimary("u"))

        type_description = compute_builtin_type_clang(builtin, prototype_mask,
                                                      type_spec, lmul)
        builtin_list.append(InstantiatedBuiltin(lmul, type_spec, full_name + "_mask",
            type_description, flags, builtin, prototype_mask,
            masked = True, index_of_mask = index_of_mask))

    return builtin_list


def instantiate_builtins(j):
    all_builtins = j["!instanceof"]["EPIBuiltin"]
    result = []
    for builtin_instance in all_builtins:
        builtin = j[builtin_instance]
        prototype = parse_prototype_seq(builtin["Prototype"])
        for lmul in IMPLEMENTED_LMULS:
            if lmul in builtin["LMUL"]:
                for type_spec in builtin["TypeRange"]:
                    result += compute_single_builtin_defs(builtin, prototype, type_spec, lmul)
    # Remove legitimate repeated instances do not have to do this every time
    builtin_set = {}
    error = False

    unique_result = []
    for b in result:
        if b.full_name in builtin_set :
            if builtin_set[b.full_name] != str(b):
                logging.error("Builtin '{}' has already been defined as '{}' but this one is '{}'".format(b.full_name, str(builtin_set[b.full_name]), str(b)))
                error = True
        else:
            builtin_set[b.full_name] = str(b)
            unique_result.append(b)

    if error:
        raise Exception("Errors found while instantiating the builtins")

    return unique_result


def emit_builtins_def(out_file, j):
    # Ideally we should not generate repeated builtins but some
    # masked operations overlap while ranging over LMUL and they generate
    # repeated builtins.
    # We use an OrderedDict as a form of OrderedSet. This is not great
    # but avoids using external packages or having to roll our own
    # data-structure.

    error = False

    inst_builtins = instantiate_builtins(j)

    out_file.write("""\
#if defined(BUILTIN) && !defined(EPI_BUILTIN)
#define EPI_BUILTIN(ID, TYPE, ATTRS) BUILTIN(ID, TYPE, ATTRS)
#endif

""")

    for b in inst_builtins:
        out_file.write("{}\n".format(b))

    out_file.write("""
#undef BUILTIN
#undef EPI_BUILTIN
""")


def emit_codegen(out_file, j):
    inst_builtins = instantiate_builtins(j)

    code_set = {}
    for b in inst_builtins:
        code_case = ""

        if b.builtin["CodegenSetID"] != 0:
            if not b.masked and b.builtin["IntrinsicName"]:
                code_case += "  ID = Intrinsic::epi_{};\n".format(b.builtin["IntrinsicName"]);
            if b.masked and b.builtin["IntrinsicNameMask"]:
                code_case += "  ID = Intrinsic::epi_{};\n".format(b.builtin["IntrinsicNameMask"]);

        if b.builtin["HasManualCodegen"] != 0:
            s = ""
            if b.masked:
                s += b.builtin["ManualCodegenMask"]
            else:
                s += b.builtin["ManualCodegen"]
            if not s.endswith("\n"):
                s += "\n"
            code_case += s;
        elif b.builtin["IntrinsicTypes"]:
            code_case += "  IntrinsicTypes = { ";
            intr_types = []
            source_intrinsic_types = b.builtin["IntrinsicTypes"][:]
            if b.masked:
                if b.builtin["HasMergeOperand"]:
                    # Skew the operands
                    for i in range(0, len(source_intrinsic_types)):
                        if source_intrinsic_types[i] >= 0:
                            source_intrinsic_types[i] += 1

            for t in source_intrinsic_types:
                if t == -1:
                    intr_types.append("ResultType");
                elif t == -2:
                    if b.masked:
                        intr_types.append("Ops[{}]->getType()".format(b.index_of_mask - 1));
                else:
                    if b.masked and t == b.index_of_mask:
                        raise Exception("Cannot refer to the mask operand like a regular operand")
                    intr_types.append("Ops[{}]->getType()".format(t));
            code_case += ", ".join(intr_types)
            code_case += " };\n";

        code_case += "  break;\n"

        if code_case not in code_set:
            code_set[code_case] = [b.full_name]
        else:
            code_set[code_case].append(b.full_name)

    for (code, cases) in code_set.items():
        for case in cases:
            out_file.write("case RISCV::EPI_BI_{}:\n".format(case))
        out_file.write(code)

def adjust_text(text):
    text = text.strip("\n")
    text = text.rstrip()
    lines = text.splitlines()
    min_leading_spaces = None
    for l in lines:
        # Blank lines are handled like newlines
        if l.strip() == "":
            continue

        leading_spaces = 0
        for c in l:
            if c == ' ':
                leading_spaces += 1
            else:
                break
        if min_leading_spaces is None:
            min_leading_spaces = leading_spaces
        else:
            min_leading_spaces = min(min_leading_spaces, leading_spaces)

    # Remove leading spaces
    if min_leading_spaces is not None:
        new_lines = []
        for l in lines:
            if l.strip() == "":
                new_lines.append("")
            else:
                new_lines.append(l[min_leading_spaces:])
        text = "\n".join(new_lines)

    return text


def format_code(source, clang_format):
    format_pipe = subprocess.Popen([clang_format, "--style=LLVM"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    format_pipe.stdin.write(source.encode())
    (outdata, errdata) = format_pipe.communicate()
    if format_pipe.returncode != 0:
        raise Exception("Call to clang-format failed\n{}".format(errdata))
    return outdata

LETTERS = [ chr(x) for x in range(ord('A'), ord('Z') + 1) ]

def emit_asciidoc_document(out_file, j, clang_format):
    inst_builtins = instantiate_builtins(j)

    all_docs = j["!instanceof"]["DocEPIBuiltin"]
    documented = set([])
    categories = {}
    for doc_record_name in all_docs:
        doc = j[doc_record_name]
        if doc["Undocumented"] != 0:
            builtin_record_name = doc["Builtin"]["printable"]
            documented.add(builtin_record_name)
            continue
        category_id = doc["Category"]["printable"]
        if category_id not in categories:
            categories[category_id] = [doc]
        else:
            categories[category_id].append(doc)

    sorted_categories = list(categories.keys())
    sorted_categories.sort(key = lambda x : j[x]["Index"])

    category_number = 0
    for category_id in sorted_categories:
        category_name = j[category_id]["Name"]
        out_file.write("=== {}\n\n".format(category_name))
        sorted_docs = categories[category_id]
        sorted_docs.sort(key = lambda x : x["Builtin"]["printable"])
        builtin_number = 0
        for doc in sorted_docs:
            builtin_record_name = doc["Builtin"]["printable"]
            builtin_record = j[builtin_record_name]

            if builtin_record_name in documented:
                logging.error("Builtin {} documented more than once".format(builtin_record_name))
                runtime_error = True
            documented.add(builtin_record_name)

            builtin_number += 1
            out_file.write("==== {}\n\n".format(doc["Title"]))
            out_file.write("Description::\n+\n--\n{}\n--\n\n".format(doc["Description"]))

            if doc["Instruction"]:
                out_file.write("Instruction::\n")
                out_file.write("[source,text]\n");
                out_file.write("----\n")
                out_file.write(doc["Instruction"] + "\n")
                out_file.write("----\n\n")

            # Unmasked
            # FIXME - Build an index instead of traversing all the builtins
            prototypes = ""
            for inst in inst_builtins:
                if inst.builtin is builtin_record and not inst.masked:
                    prototypes += inst.c_prototype(doc["ParameterNames"]) + "\n";
            formatted_prototypes = format_code(prototypes, clang_format)

            out_file.write("Prototypes::\n")
            out_file.write("[source,cpp]\n");
            out_file.write("----\n")
            out_file.write(formatted_prototypes.decode())
            out_file.write("----\n\n")

            if doc["Operation"]:
                out_file.write("Operation::\n")
                out_file.write("[source,text]\n");
                out_file.write("----\n")
                out_file.write(adjust_text(doc["Operation"]) + "\n")
                out_file.write("----\n\n")

            # Masked
            if builtin_record["HasMask"] != 0:
                out_file.write("Masked prototypes::\n")
                prototypes = ""
                for inst in inst_builtins:
                    if inst.builtin is builtin_record and inst.masked:
                        prototypes += inst.c_prototype(doc["ParameterNames"]) + "\n";
                formatted_prototypes = format_code(prototypes, clang_format)

                out_file.write("[source,cpp]\n");
                out_file.write("----\n")
                out_file.write(formatted_prototypes.decode())
                out_file.write("----\n\n")

                if doc["OperationMask"]:
                    out_file.write("Masked operation::\n")
                    out_file.write("[source,text]\n");
                    out_file.write("----\n")
                    out_file.write(adjust_text(doc["OperationMask"]) + "\n")
                    out_file.write("----\n\n")
        category_number += 1

    for builtin in j["!instanceof"]["EPIBuiltin"]:
        if builtin not in documented:
            logging.warning("Builtin '{}' has not been documented".format(builtin));

def emit_tests(out_file, j):
    out_file.write(r"""// RUN: %clang --target=riscv64-unknown-linux-gnu -mepi -S -emit-llvm -O2 -o - %s \
// RUN:       | FileCheck --check-prefix=CHECK-O2 %s

""")
    inst_builtins = instantiate_builtins(j)
    already_emitted = set([])
    for b in inst_builtins:
        # NOTE: this if would skip the builtins that are added to the
        # compatibility header. But since at the moment we are still registering
        # them in clang, that's probably better to include them also in this test
        #if b.builtin["EPIRVVHeader"] == 1:
        #    continue

        # FIXME:
        if b.lmul not in [1, 2, 4]:
            continue

        # The current strategy does not work for builtins that expect
        # constant expressions. So skip these ones and let them be tested
        # elsewhere
        if "K" in b.builtin["Prototype"]:
            continue

        # This is a bit of stupid heuristic to skip builtins we know
        # don't work.
        if b.builtin["HasManualCodegen"] != 0:
            codegen = b.builtin["ManualCodegen"] \
                    if not b.masked else b.builtin["ManualCodegenMask"]
            if "ErrorUnsupported" in codegen:
                continue

        if b.full_name in already_emitted:
            continue

        already_emitted.add(b.full_name)

        (return_type, parameter_types) = b.c_prototoype_items()

        parameters = []
        arguments = []
        i = 0
        for p in parameter_types:
            arg = "arg_{}".format(i)
            i += 1
            parameters.append("{} {}".format(p, arg))
            arguments.append(arg)

        out_file.write("{} test_{}({})\n{{\n    return __builtin_epi_{}({});\n}}\n\n".format(
            return_type,
            b.full_name,
            ", ".join(parameters),
            b.full_name,
            ", ".join(arguments)))

def single_test_clang(clang, tmp_file, extra_flags = []):
    DEVNULL = open(os.devnull, 'wb')
    try:
        subprocess.check_call([clang,
            "--target=riscv64-unknown-linux-gnu",
            "-fno-crash-diagnostics",
            "-mepi",
            "-S",
            "-o", "/dev/null",
            "-x", "c",
            tmp_file.name] + extra_flags,
            stdout=DEVNULL,
            stderr=DEVNULL)
    except Exception as e:
        # print "ERROR: {}".format(e.output)
        return False
    return True

def emit_compatibility_header(out_file, j):
    inst_builtins = instantiate_builtins(j)

    out_file.write("""\
#ifndef __EPI_RVV_H
#define __EPI_RVV_H

#include <riscv_vector.h>

""")

    ELEN = 64 # In EPI, ELEN=64
    # Vector types
    out_file.write("// Vector types\n")
    vector_types = "#define __epi_${VScale}x${Type} v${ExtendedType}${LMul}_t"
    for lm in IMPLEMENTED_LMULS:
        # We only have LMUL >= 1, so no need to check if we need to prepend 'm' or 'mf'
        lmul = "m" + str(lm)
        for type in ['i8', 'i16', 'i32', 'i64', 'f32', 'f64']:
            ex_type = "int" + type[1:] if type[0] == 'i' else "float" + type[1:]
            size = int(type[1:])
            vscale = int(ELEN * lm / size)
            out_file.write("{}\n".format(string.Template(vector_types).substitute(
                ExtendedType=ex_type, LMul=lmul, VScale=vscale, Type=type)))

    out_file.write("\n")
    # Mask type
    out_file.write("// Mask types\n")
    mask_types = "#define __epi_${VScale}xi1 vbool${N}_t"
    for vscale in [1, 2, 4, 8, 16, 32, 64]:
        n = int(ELEN/vscale)
        out_file.write("{}\n".format(string.Template(mask_types).substitute(
            N = n, VScale = vscale)))

    out_file.write("\n")
    # Builtins
    out_file.write("// Builtin mappings\n")
    for ib in inst_builtins:
        if ib.builtin["EPIRVVHeader"] == 1:
            if not ib.masked:
                code = ib.builtin["CompatibilityCode"]
            else:
                code = ib.builtin["CompatibilityCodeMasked"]
            if "WidenedLMul" in code and ib.lmul > 4:
                continue # We filter out undefined values of LMul
            code = string.Template(code)
            subs = {}
            subs["FullName"] = ib.full_name
            subs["Name"] = "__riscv_" + ib.builtin["Name"]
            # At the moment vlseg and vsseg builtins are not implemented upstream
            # Once they are, uncommenitng this 'if' statement should be enough to make them work
            #if subs["Name"].startswith("vlseg") or subs["Name"].startswith("vsseg"):
            #    subs["Tuple"] = subs["Name"][5]
            subs["Type"] = TypePrimary(ib.type_spec).render_for_name()
            subs["Size"] = subs["Type"][1:]
            subs["UType"] = "u" + subs["Size"]
            subs["WidenedSize"] = str(2 * int(subs["Size"]))
            subs["WidenedType"] = subs["Type"][0] + subs["WidenedSize"]
            subs["WidenedUType"] = "u" + subs["WidenedSize"]
            # We only have LMUL >= 1, so no need to check if we need to prepend 'm' or 'mf'
            subs["LMul"] = "m" + str(ib.lmul)
            subs["WidenedLMul"] = "m" + str(2 * ib.lmul)
            subs["Boolean"] = int(int(subs["Size"]) / ib.lmul)
            if "vred" in ib.full_name or "vfred" in ib.full_name:
                if ib.lmul == 1:
                    subs["LMulExt"] = subs["LMulTrunc"] = subs["End"] = ""
                else:
                    subs["LMulExt"] = "__riscv_vlmul_ext_v_" + subs["Type"] + "m1_" + subs["Type"] + subs["LMul"] + "("
                    subs["LMulTrunc"] = "__riscv_vlmul_trunc_v_" + subs["Type"] + subs["LMul"] + "_" + subs["Type"] + "m1" + "("
                    subs["End"] = ")"
            if "slide" in ib.full_name:
                subs["ifFloat"] = "f" if subs["Type"][0] == "f" else ""
                subs["Reg"] = "f" if subs["Type"][0] == "f" else "x"
            out_file.write("{}\n".format(code.substitute(subs)))

    out_file.write("""
#endif //__EPI_RVV_H
""")

def emit_header_tests(out_file, j):
    out_file.write(r"""// RUN: %clang -ffreestanding --target=riscv64-unknown-linux-gnu -mepi -S -emit-llvm -O2 -o - %s \
// RUN:       | FileCheck --check-prefix=CHECK-O2 %s

""")
    out_file.write("""#include <epi_rvv.h>\n\n""")
    inst_builtins = instantiate_builtins(j)
    already_emitted = set([])
    for b in inst_builtins:
        if b.builtin["EPIRVVHeader"] == 0:
            continue

        # FIXME:
        if b.lmul not in [1, 2, 4]:
            continue

        # The current strategy does not work for builtins that expect
        # constant expressions. So skip these ones and let them be tested
        # elsewhere
        if "K" in b.builtin["Prototype"]:
            continue

        # This is a bit of stupid heuristic to skip builtins we know
        # don't work.
        if b.builtin["HasManualCodegen"] != 0:
            codegen = b.builtin["ManualCodegen"] \
                    if not b.masked else b.builtin["ManualCodegenMask"]
            if "ErrorUnsupported" in codegen:
                continue

        if b.full_name in already_emitted:
            continue

        already_emitted.add(b.full_name)

        (return_type, parameter_types) = b.c_prototoype_items()

        parameters = []
        arguments = []
        i = 0
        for p in parameter_types:
            arg = "arg_{}".format(i)
            i += 1
            parameters.append("{} {}".format(p, arg))
            arguments.append(arg)

        out_file.write("{} test_{}({})\n{{\n    return __builtin_epi_{}({});\n}}\n\n".format(
            return_type,
            b.full_name,
            ", ".join(parameters),
            b.full_name,
            ", ".join(arguments)))

def emit_header_verify(out_file, j):
    out_file.write(r"""// RUN: %clang -ffreestanding -Xclang -verify --target=riscv64-unknown-linux-gnu -mepi -S -emit-llvm -O2 -o - %s

""")
    out_file.write("""// expected-no-diagnostics\n\n""")
    out_file.write("""#include <epi_rvv.h>\n\n""")
    inst_builtins = instantiate_builtins(j)
    already_emitted = set([])
    for b in inst_builtins:
        if b.builtin["EPIRVVHeader"] == 0:
            continue

        # FIXME:
        if b.lmul not in [1, 2, 4]:
            continue

        # The current strategy does not work for builtins that expect
        # constant expressions. So skip these ones and let them be tested
        # elsewhere
        if "K" in b.builtin["Prototype"]:
            continue

        # This is a bit of stupid heuristic to skip builtins we know
        # don't work.
        if b.builtin["HasManualCodegen"] != 0:
            codegen = b.builtin["ManualCodegen"] \
                    if not b.masked else b.builtin["ManualCodegenMask"]
            if "ErrorUnsupported" in codegen:
                continue

        if b.full_name in already_emitted:
            continue

        already_emitted.add(b.full_name)

        (return_type, parameter_types) = b.c_prototoype_items()

        parameters = []
        arguments = []
        i = 0
        for p in parameter_types:
            arg = "arg_{}".format(i)
            i += 1
            parameters.append("{} {}".format(p, arg))
            arguments.append(arg)

        out_file.write("{} test_{}({})\n{{\n    return __builtin_epi_{}({});\n}}\n\n".format(
            return_type,
            b.full_name,
            ", ".join(parameters),
            b.full_name,
            ", ".join(arguments)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate instruction table")
    parser.add_argument("--mode", required=True,
            choices=["builtins-def", "codegen", "tests", "docs", "compat-header", "compat-header-tests", "compat-header-verify"], help="Mode of operation")
    parser.add_argument("--tablegen", required=True, help="Path of tablegen")
    parser.add_argument("--output-file", required=False, help="Output file. stdout otherwise")
    parser.add_argument("--clang-format", required=False, help="Path of clang-format")
    parser.add_argument("-I", dest="include_paths", required=False, default=[],
                      help="Include path", action="append")
    parser.add_argument("input_tblgen", help="File with the tablegen description")
    args = parser.parse_args()

    out_file = sys.stdout
    if args.output_file:
        out_file = open(args.output_file, "w")

    j = process_tblgen_file(args.tablegen, args.input_tblgen, args.include_paths)

    if args.mode == "builtins-def":
        emit_builtins_def(out_file, j)
    elif args.mode == "codegen":
        emit_codegen(out_file, j)
    elif args.mode == "docs":
        if not args.clang_format:
            parser.error("You have to pass --clang-format when generating documentation")
        emit_asciidoc_document(out_file, j, args.clang_format)
    elif args.mode == "tests":
        emit_tests(out_file, j)
    elif args.mode == "compat-header":
        emit_compatibility_header(out_file, j)
    elif args.mode == "compat-header-tests":
        emit_header_tests(out_file, j)
    elif args.mode == "compat-header-verify":
        emit_header_verify(out_file, j)
    else:
        raise Exception("Unexpected mode '{}".format(args.mode))

    out_file.close()
    if runtime_error:
        sys.exit(1)
