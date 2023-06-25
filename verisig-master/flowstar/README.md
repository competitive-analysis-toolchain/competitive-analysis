This is a modification of flowstar. More precisely, it is a modification of
the flowstar shipped with Verisig. The GPL3 license, therefore, still applies.

# Dependencies
* bison (newer than OS X default, e.g. via homebrew)
* flex (newer than OS X default)
* mpfr

For bison and flex, if you are using Homebrew, installation is not sufficient.
We recommend either forcing to link them or following the official Homebrew
instructions to add a line in your ./zshrc to make it be found before the
default OS X version.

# Changelog
* The makefile has been adapted so that it works for OS X (on M1 macs) with
  libraries installed via Homebrew
* The use of bits/stdc++ in DNNResets.cpp has been changed to the more
  specific use of unordered_sets
