#!/bin/bash

# Comprueba que las imagenes resultantes son las esperadas

diff out.pgm expected_output/out_original.pgm
diff out_hsl.ppm expected_output/out_hsl_original.ppm
diff out_yuv.ppm expected_output/out_yuv_original.ppm
