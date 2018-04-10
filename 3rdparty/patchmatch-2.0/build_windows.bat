@rem It works without /nodefaultlib:libcmt.lib

@rem Optimized (slow to build)
@rem call mex OPTIMFLAGS="/Ox /Oi /Oy /DNDEBUG /fp:fast /arch:SSE2 /DMEX_MODE /openmp"  mexutil.cpp nn.cpp nnmex.cpp patch.cpp vecnn.cpp simnn.cpp allegro_emu.cpp -output nnmex
@rem call mex OPTIMFLAGS="/Ox /Oi /Oy /DNDEBUG /fp:fast /arch:SSE2 /DMEX_MODE /openmp"  mexutil.cpp nn.cpp votemex.cpp patch.cpp vecnn.cpp simnn.cpp allegro_emu.cpp -output votemex

@rem Unoptimized (fast to build)
call mex OPTIMFLAGS="/DNDEBUG /DMEX_MODE /openmp"  knn.cpp mexutil.cpp nn.cpp nnmex.cpp patch.cpp vecnn.cpp simnn.cpp allegro_emu.cpp -output nnmex
call mex OPTIMFLAGS="/DNDEBUG /DMEX_MODE /openmp"  knn.cpp mexutil.cpp nn.cpp votemex.cpp patch.cpp vecnn.cpp simnn.cpp allegro_emu.cpp -output votemex
