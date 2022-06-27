#!/usr/bin/env python3

import sys
import subprocess

# This script expects a list of arguments separated by --emulator
SPLITTING_FLAG="--emulator"

if SPLITTING_FLAG not in sys.argv:
    raise Exception("Flag --emulator is missing")

compiler_command = []
emulator_command = []

argv = sys.argv[1:]
l = compiler_command
for arg in argv:
    if arg != SPLITTING_FLAG:
        l.append(arg)
    else:
        l = emulator_command

compiler_command.append("-###")
driver_invocation = emulator_command + compiler_command

# print("{}".format(" ".join(driver_invocation)))

# FIXME: is utf-8 a sensible option here?
driver = subprocess.run(driver_invocation, \
         stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
         encoding="utf-8")

frontend_invocation = ""
for line in driver.stderr.splitlines():
    if line[0] == ' ' and '"-fc1"' in line:
        frontend_invocation = line
        break

if not frontend_invocation:
    raise Exception("Could not identify command invocation")

# print("{}".format(frontend_invocation))

frontend_invocation = " ".join(emulator_command) + " " + frontend_invocation

# FIXME: relies on the driver printing a properly quoted string
subprocess.run(frontend_invocation, shell=True, check=True)
