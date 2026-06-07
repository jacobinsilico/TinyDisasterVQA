cd /app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best

source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
export TILER_INTEGRAL_GENERATOR_PATH=/app/install/gap9-sdk/tools/autotiler_v3/Generators/IntegralImage
export TILER_CNN_GENERATOR_PATH=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Generators
export TILER_INTEGRAL_KERNEL_PATH=/app/install/gap9-sdk/tools/autotiler_v3/Generators/IntegralImage
export TILER_ISP_GENERATOR_PATH=/app/install/gap9-sdk/tools/autotiler_v3/ISP_Generators
export TILER_FFT2D_TWIDDLE_PATH=/app/install/gap9-sdk/tools/autotiler_v3/Generators/FFT2DModel
export TILER_CNN_GENERATOR_PATH_SQ8=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Generators_SQ8
export TILER_BILINEAR_RESIZE_KERNEL_PATH=/app/install/gap9-sdk/tools/autotiler_v3/ISP_Libraries
export TILER_LIB=/app/install/gap9-sdk/tools/autotiler_v3/Autotiler/LibTile.a
export TILER_FFT2D_GENERATOR_PATH=/app/install/gap9-sdk/tools/autotiler_v3/Generators/FFT2DModel
export TILER_CNN_KERNEL_PATH_SQ8=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Libraries_SQ8
export TILER_DSP_GENERATOR_PATH=/app/install/gap9-sdk/tools/autotiler_v3/DSP_Generators
export TILER_CNN_KERNEL_PATH_FP16X=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Libraries_fp16/CNN_Libraries_fp16x
export TILER_INC=/app/install/gap9-sdk/tools/autotiler_v3/Autotiler
export TILER_CNN_KERNEL_PATH=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Libraries
export AT_HOME=/app/install/gap9-sdk/tools/autotiler_v3
export TILER_CNN_KERNEL_PATH_NE16=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Libraries_NE16
export TILER_GENERATOR_PATH=/app/install/gap9-sdk/tools/autotiler_v3/Generators
export TILER_CNN_KERNEL_PATH_FP16=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Libraries_fp16
export TILER_ISP_KERNEL_PATH=/app/install/gap9-sdk/tools/autotiler_v3/ISP_Libraries
export TILER_PATH=/app/install/gap9-sdk/tools/autotiler_v3
export TILER_CNN_GENERATOR_PATH_FP16=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Generators_fp16
export TILER_BILINEAR_RESIZE_GENERATOR_PATH=/app/install/gap9-sdk/tools/autotiler_v3/ISP_Generators
export TILER_DSP_KERNEL_PATH=/app/install/gap9-sdk/tools/autotiler_v3/DSP_Libraries
export TILER_DSP_KERNEL_V2_PATH=/app/install/gap9-sdk/tools/autotiler_v3/DSP_Librariesv2
export TILER_FFT2D_KERNEL_PATH=/app/install/gap9-sdk/tools/autotiler_v3/Generators/FFT2DModel
export TILER_TWID_GEN_SCRIPT=/app/install/gap9-sdk/tools/autotiler_v3/DSP_Librariesv2/TransformFunctions/GenTwid/GenTwid.py
export TILER_CNN_GENERATOR_PATH_NE16=/app/install/gap9-sdk/tools/autotiler_v3/CNN_Generators_NE16
export TILER_EMU_INC=/app/install/gap9-sdk/tools/autotiler_v3/Emulation
export CCACHE_BASEDIR=/app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best
mkdir -p /app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best/build
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi
(cd /app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best; cmake -B build -G"Unix Makefiles")
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi
(cd /app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best; cmake --build build --target clean)
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi
(cd /app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best; cmake --build build --target all -j 12 )
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi
(cd /app/TinyDisasterVQA/gap9_generated/tdm_xxs_single_ce_224_best; rm -f build/S*ArgName*ItemSize*Dim*.dat; cmake --build build --target run )
exit $?
