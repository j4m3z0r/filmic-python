# filmic-python: Filmic Tone Mapping for Python

This is essentially a line-for-line port of John Hable's
[fw-public](https://github.com/johnhable/fw-public) C++ code. His [blog
post](http://filmicworlds.com/blog/filmic-tonemapping-with-piecewise-power-curves/)
goes into a lot more detail on what this is all about.

The original C++ code is likely much more suitable for use in actual
applications; this is mostly useful for situations where you want to bake a
tone-curve in your build process, and don't want to have to maintain a build
process for your build process. It might also be useful to integrate into tools
that expose a Python API, like Blender.

This is released under the same Creative Commons CCO license as the original.
Complete license terms in LICENSE.txt.

You will mostly want to interact with the FilmicColorGrading module, and a
simple main.py is included that shows how to invoke it. main.py will write a
CSV file of mappings from linear to tone-mapped values.

