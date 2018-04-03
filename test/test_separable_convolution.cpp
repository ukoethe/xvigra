/************************************************************************/
/*                                                                      */
/*     Copyright 2017-2018 by Ullrich Koethe                            */
/*                                                                      */
/*    This file is part of the XVIGRA image analysis library.           */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#include "unittest.hpp"

// #ifndef NDEBUG
// #define XVIGRA_CHECK_BOUNDS
// #endif

#include <xvigra/separable_convolution.hpp>
#include <xvigra/array_nd.hpp>
#include <xvigra/image_io.hpp>
// #include "xsimd/config/xsimd_instruction_set.hpp"

// using info_map_type = std::map<int, std::string>;

// info_map_type init_instruction_map()
// {
//     info_map_type res;
//     res[XSIMD_X86_SSE_VERSION] = "Intel SSE";
//     res[XSIMD_X86_SSE2_VERSION] = "Intel SSE2";
//     res[XSIMD_X86_SSE3_VERSION] = "Intel SSE3";
//     res[XSIMD_X86_SSSE3_VERSION] = "Intel SSSE3";
//     res[XSIMD_X86_SSE4_1_VERSION] = "Intel SSE4.1";
//     res[XSIMD_X86_SSE4_2_VERSION] = "Intel SSE4.2";
//     res[XSIMD_X86_AVX_VERSION] = "Intel AVX";
//     res[XSIMD_X86_FMA3_VERSION] = "Intel FMA3";
//     res[XSIMD_X86_AVX2_VERSION] = "Intel AVX2";
//     res[XSIMD_X86_MIC_VERSION] = "Intel MIC";
//     res[XSIMD_X86_AMD_SSE4A_VERSION] = "AMD SSE4A";
//     res[XSIMD_X86_AMD_FMA4_VERSION] = "AMD FMA4";
//     res[XSIMD_X86_AMD_XOP_VERSION] = "AMD XOP";
//     res[XSIMD_PPC_VMX_VERSION] = "PowerPC VM";
//     res[XSIMD_PPC_VSX_VERSION] = "PowerPC VSX";
//     res[XSIMD_PPC_QPX_VERSION] = "PowerPC QPX";
//     res[XSIMD_ARM_NEON_VERSION] = "ARM Neon";
//     res[XSIMD_VERSION_NUMBER_NOT_AVAILABLE] = "No SIMD available";
//     return res;
// }

// std::string get_instruction_set_name()
// {
//     static info_map_type info_map(init_instruction_map());
//     return info_map[XSIMD_INSTR_SET];
// }

namespace xvigra
{
    TEST(separable_convolution, 3d_average_filter)
    {
        auto && kernel = averaging_kernel_1d<float>(1);
        // array_nd<float, 3> in({5, 5, 5}, 1),
        array_nd<float, 3> in({100, 200, 300}, 1),
                           out(in.shape(), 0);
        slow_separable_convolution(in, out, kernel);
        EXPECT_TRUE(allclose(out, 1.0f));
        out = 0.0f;
        separable_convolution(in, out, kernel);
        EXPECT_TRUE(allclose(out, 1.0f));
        if(out.size() < 200)
        {
            std::cerr << out << "\n";
        }
    }

    TEST(separable_convolution, 2d_gauss_filter)
    {
        auto && kernel = gaussian_kernel_1d<float>(2.0);
        array_nd<float> in = read_image("color_image.tif"),
                        out(in.shape(), 0);
        separable_convolution(2_d, in, out, kernel);
        write_image("smooth.png", out);
    }
} // namespace xvigra
