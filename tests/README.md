# Release validation

Model tests will accompany the implementation. Planned checks cover strict
checkpoint loading, source order, specialist target index, conditioned encoder
loading, dataset split consistency, and a real inference example.

Training release checks must exercise a finite update and checkpoint resume in
the documented environment. Synthetic architecture compatibility alone does
not establish that a released checkpoint or complete workflow works.

This initial repository contains no model test suite.
