### 3.12 The argument parser

Defines the command line interface, so all available flags are documented here and the batch cell
below can call `main(argv)` with the same defaults.

The `if __name__ == "__main__"` guard from the original file is deliberately absent. In a notebook
`__name__` is `"__main__"`, so it would run against the kernel's own arguments.
