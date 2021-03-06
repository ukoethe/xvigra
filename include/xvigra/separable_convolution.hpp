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

#ifndef XVIGRA_SEPARABLE_CONVOLUTION_HPP
#define XVIGRA_SEPARABLE_CONVOLUTION_HPP

#include <deque>

#ifdef XVIGRA_USE_SIMD
#  include <xsimd/xsimd.hpp>
#endif

#include "padding.hpp"
#include "slice.hpp"
#include "array_nd.hpp"
#include "functor_base.hpp"
#include "kernel.hpp"

namespace xvigra
{
    /***********************/
    /* convolution_options */
    /***********************/

    struct convolution_options
    {
        using padding_vec = tiny_vector<padding_mode>;

        bool simd = true;
        padding_vec left_padding{reflect_padding}, right_padding{reflect_padding};

        convolution_options & use_simd(bool v=true)
        {
            simd = v;
            return *this;
        }

        convolution_options & padding(padding_mode p)
        {
            return padding(p, p);
        }

        convolution_options & padding(padding_mode left, padding_mode right)
        {
            left_padding = left;
            right_padding = right;
            return *this;
        }

        template <index_t N1, index_t N2>
        convolution_options & padding(tiny_vector<padding_mode, N1> const & left,
                                      tiny_vector<padding_mode, N2> const & right)
        {
            left_padding = left;
            right_padding = right;
            return *this;
        }

        padding_mode get_left_padding(index_t d) const
        {
            if(left_padding.size() == 0)
            {
                return reflect_padding;
            }
            if(left_padding.size() == 1)
            {
                return left_padding[0];
            }
            vigra_precondition(0 <= d && d < left_padding.size(),
                "convolution_options.get_left_padding(d): requested dimension out of bounds.");
            return left_padding[d];
        }

        padding_mode get_right_padding(index_t d) const
        {
            if(right_padding.size() == 0)
            {
                return reflect_padding;
            }
            if(right_padding.size() == 1)
            {
                return right_padding[0];
            }
            vigra_precondition(0 <= d && d < right_padding.size(),
                "convolution_options.get_right_padding(d): requested dimension out of bounds.");
            return right_padding[d];
        }
    };

    /******************************/
    /* slow_separable_convolution */
    /******************************/

    namespace detail
    {

        template <class T1, index_t N1, class T2, index_t N2, class Kernel>
        inline void convolve_row(view_nd<T1, N1> in, view_nd<T2, N2> out, Kernel const & kernel)
        {
            for(index_t j=0; j<out.shape(0); ++j)
            {
                out(j) = in(j)*kernel(0);
            }
            for(index_t l=1; l<kernel.shape(0); ++l)
            {
                for(index_t j=0; j<out.shape(0); ++j)
                {
                    out(j) += in(j+l)*kernel(l);
                }
            }

        }

    } // namespace detail

    template <class T1, index_t N1, class T2, index_t N2, class T3>
    void slow_separable_convolution(view_nd<T1, N1> in, view_nd<T2, N2> out,
                                    kernel_1d<T3> const & kernel,
                                    convolution_options const & options = convolution_options())
    {
        using namespace slicing;
        auto rev_kernel = kernel.view(slice(_,_,-1));
        index_t right = kernel.center(),
                left  = kernel.size() - right - 1;

        index_t N = in.dimension();

        slicer nav(in.shape());
        {
            // operate on last dimension first
            nav.set_free_axes(N-1);
            padding_mode left_padding  = options.get_left_padding(N-1),
                         right_padding = options.get_right_padding(N-1);
            array_nd<T2, 1> padded(shape_t<1>{in.shape(N-1)+left+right});
            for(; nav.has_more(); ++nav)
            {
                copy_with_padding(in.view(*nav), padded, left_padding, left, right_padding, right);
                detail::convolve_row(padded, out.view(*nav), rev_kernel);
            }
        }

        for( index_t d = N-2; d >= 0; --d )
        {
            // operate on further dimensions
            nav.set_free_axes(d);
            padding_mode left_padding  = options.get_left_padding(d),
                         right_padding = options.get_right_padding(d);
            array_nd<T2, 1> padded(shape_t<1>{out.shape(d)+left+right});
            for(; nav.has_more(); ++nav)
            {
                copy_with_padding(out.view(*nav), padded, left_padding, left, right_padding, right);
                detail::convolve_row(padded, out.view(*nav), rev_kernel);
            }
        }
    }

    /*********************************/
    /* separable_convolution_functor */
    /*********************************/

    namespace detail
    {

    #if XVIGRA_USE_SIMD
        template <class T,
                  VIGRA_REQUIRE<std::is_floating_point<T>::value>>
        inline void simd_mul_row(T const * src, index_t size, T * dest, T a)
        {
            auto ba = xsimd::set_simd(a);

            constexpr index_t simd_size = xsimd::simd_batch_traits<decltype(ba)>::size;
            constexpr std::size_t align_bits = xsimd::simd_batch_traits<decltype(ba)>::align - 1;

            for(; size>0; --size, ++dest, ++src)
            {
                if(((std::size_t)dest & align_bits) == 0)
                {
                    break; // dest is SIMD-aligned
                }
                *dest = *src * a;
            }

            index_t simd_end = size - size % simd_size;
            for(index_t j=0; j<simd_end; j += simd_size)
            {
                (ba * xsimd::load_unaligned(src+j)).store_aligned(dest+j);
            }
            for(index_t j=simd_end; j<size; ++j)
            {
                *(dest+j) = *(src+j) * a;
            }
        }

        template <class T,
                  VIGRA_REQUIRE<std::is_floating_point<T>::value>>
        inline void simd_fma_row(T const * src, index_t size, T * dest, T a)
        {
            auto ba = xsimd::set_simd(a);

            constexpr index_t simd_size = xsimd::simd_batch_traits<decltype(ba)>::size;
            constexpr std::size_t align_bits = xsimd::simd_batch_traits<decltype(ba)>::align - 1;

            for(; size>0; --size, ++dest, ++src)
            {
                if(((std::size_t)dest & align_bits) == 0)
                {
                    break; // dest is SIMD-aligned
                }
                *dest += *src * a;
            }

            index_t simd_end = size - size % simd_size;
            for(index_t j=0; j<simd_end; j += simd_size)
            {
                xsimd::fma(ba, xsimd::load_unaligned(src+j), xsimd::load_aligned(dest+j)).store_aligned(dest+j);
            }
            for(index_t j=simd_end; j<size; ++j)
            {
                *(dest+j) += *(src+j) * a;
            }
        }
    #else
        template <class T1, class T2, class T3>
        inline void simd_mul_row(T1 const * src, index_t size, T2 * dest, T3 a)
        {
            vigra_fail("internal error: invalid call to SIMD function.");
        }

        template <class T1, class T2, class T3>
        inline void simd_fma_row(T1 const * src, index_t size, T2 * dest, T3 a)
        {
            vigra_fail("internal error: invalid call to SIMD function.");
        }
    #endif

    } // namespace detail

        // introduction of convolve_columns gives a 5x speed-up
        // SIMD gives another 3x
        // taking advantage of the kernel symmetry has negligible effect
    struct separable_convolution_functor
    {
        std::string name = "separable_convolution";

        template <class E1, class E2, class ... ARGS>
        void operator()(E1 && e1, E2 && e2, ARGS ... a) const
        {
            auto && a1 = eval_expr(std::forward<E1>(e1));
            auto && a2 = eval_expr(std::forward<E2>(e2));
            impl(0, make_view(a1), make_view(a2), std::forward<ARGS>(a)...);
        }

        template <class E1, class E2, class ... ARGS>
        void operator()(dimension_hint dim, E1 && e1, E2 && e2, ARGS ... a) const
        {
            vigra_precondition((index_t)e1.dimension() == dim || (index_t)e1.dimension() == dim+1,
                name + "(): input dimension contradicts dimension_hint.");

            auto && a1 = eval_expr(std::forward<E1>(e1));
            auto && a2 = eval_expr(std::forward<E2>(e2));

            if((index_t)a1.dimension() == dim)
            {
                impl(0, make_view(a1), make_view(a2), std::forward<ARGS>(a)...);
            }
            // else if(a1.strides(dim) == 1 && a2.strides(dim) == 1)
            // {
            //     // FIXME: implement simultaneous convolution over innermost dimension and channels
            // }
            else
            {
                auto && v1 = make_view(a1);
                auto && v2 = make_view(a2);
                for(index_t k=0; k<v1.shape(dim); ++k)
                {
                    impl(0, v1.bind(dim, k), v2.bind(dim, k), std::forward<ARGS>(a)...);
                }
            }
        }

        template <class T1, index_t N1, class T2, index_t N2, class T3>
        void impl(index_t dim, view_nd<T1, N1> in, view_nd<T2, N2> out,
                  kernel_1d<T3> const & kernel,
                  convolution_options const & options = convolution_options()) const
        {
            impl(dim, std::move(in), std::move(out),
                 std::vector<kernel_1d<T3>>(in.dimension(), kernel), options);
        }

        template <class T1, index_t N1, class T2, index_t N2, class Kernels,
                  VIGRA_REQUIRE<kernel_1d_concept<typename std::decay_t<Kernels>::value_type>::value>>
        void impl(index_t dim, view_nd<T1, N1> in, view_nd<T2, N2> out,
                  Kernels && kernels,
                  convolution_options const & options = convolution_options()) const
        {
            vigra_precondition(in.shape() == out.shape(),
                name + "(): shape mismatch between input and output.");
            vigra_precondition(dim > 0 || kernels.size() == in.dimension(),
                name + "(): number of kernels doesn't match data dimension.");

            padding_mode left_padding  = options.get_left_padding(dim),
                         right_padding = options.get_right_padding(dim);

            if(in.dimension() == 1)
            {
                // execute convolution over right-most dimension
                convolve_row(in.template view<1>(), out.template view<1>(), kernels[dim],
                             options.simd, left_padding, right_padding);
            }
            else
            {
                using tmp_type = std::conditional_t<std::is_integral<T1>::value, float, T1>;
                array_nd<tmp_type> tmp(in.shape()); // FIXME: use less tmp memory
                for(index_t k=0; k<in.shape(0); ++k)
                {
                    impl(dim+1, in.bind(0,k), tmp.bind(0,k), kernels, options);
                }
                slicer nav(out.shape());
                nav.set_free_axes(shape_t<>{0, (index_t)out.dimension()-1});
                for(; nav.has_more(); ++nav)
                {
                    // execute convolution over left-most dimension, working
                    // along rows in the inner loop
                    convolve_columns(tmp.view(*nav).template view<2>(), out.view(*nav).template view<2>(),
                                     kernels[dim], options.simd, left_padding, right_padding);
                }
            }
        }

        template <class T1, class T2, class T3>
        void convolve_row(view_nd<T1, 1> && in, view_nd<T2, 1> && out,
                          kernel_1d<T3> const & kernel, bool use_simd,
                          padding_mode left_padding, padding_mode right_padding) const
        {
#ifdef XVIGRA_USE_SIMD
            use_simd = use_simd && out.is_contiguous() &&
                       std::is_same<T1, T2>::value && std::is_floating_point<T1>::value;
#else
            use_simd = false;
#endif
            using namespace slicing;
            auto rev_kernel = kernel.view(slice(_,_,-1));
            index_t right = kernel.center(),
                    left  = kernel.size() - right - 1;
            index_t start = (left_padding == no_padding) ? left : 0;
            index_t end   = (right_padding == no_padding) ? in.shape(0) - right : in.shape(0);
            if(use_simd && in.is_contiguous())
            {
                detail::simd_mul_row(&in(start), end-start, &out(start), rev_kernel(left));
            }
            else
            {
                // out.view(slice(start, end)) += rev_kernel(left)*in.view(slice(start, end));
                for(index_t l=start; l<end; ++l)
                {
                    out(l) = rev_kernel(left)*in(l+start);
                }
            }
            if(!in.is_contiguous())
            {
                array_nd<float, 1> padded(shape_t<1>{in.shape(0)+left+right});
                copy_with_padding(in, padded, left_padding, left, right_padding, right);
                for(index_t k=0; k<rev_kernel.size(); ++k)
                {
                    if(k==left)
                    {
                        continue;
                    }
                    if(use_simd)
                    {
                        detail::simd_fma_row(&padded(k+start), end-start, &out(start), rev_kernel(k));
                    }
                    else
                    {
                        // out.view(slice(start, end)) += rev_kernel(k)*padded.view(slice(k+start, k+end));
                        for(index_t l=start; l<end; ++l)
                        {
                            out(l) += rev_kernel(k)*padded(l+k+start);
                        }
                    }
                }
            }
            else
            {
                // for(index_t k=-left; k<0; ++k)
                for(index_t k=-left; k<=right; ++k)
                {
                    if(k==0)
                    {
                        continue;
                    }
                    if(start + k < 0)
                    {
                        // left border treatment
                        // invariant: left_padding != no_padding
                        if(left_padding == reflect_padding)
                        {
                            for(index_t l=0; l<-k; ++l)
                            {
                                out(l) += rev_kernel(k+left)*in(-l-k);
                            }
                        }
                        else if(left_padding == reflect0_padding)
                        {
                            for(index_t l=0; l<-k; ++l)
                            {
                                out(l) += rev_kernel(k+left)*in(-l-k-1);
                            }
                        }
                        else if(left_padding == repeat_padding)
                        {
                            for(index_t l=0; l<-k; ++l)
                            {
                                out(l) += rev_kernel(k+left)*in(0);
                            }
                        }
                        else if(left_padding == periodic_padding)
                        {
                            for(index_t l=0; l<-k; ++l)
                            {
                                out(l) += rev_kernel(k+left)*in(in.shape(0)+k+l);
                            }
                        }
                        // else if(left_padding == zero_padding) pass;

                        // convolution of interior
                        if(use_simd)
                        {
                            detail::simd_fma_row(&in(0), in.shape(0)+k, &out(-k), rev_kernel(k+left));
                        }
                        else
                        {
                            // out.view(slice(-k, in.shape(0))) += rev_kernel(k+left)*in.view(slice(0, in.shape(0)+k));
                            for(index_t l=-k; l<in.shape(0); ++l)
                            {
                                out(l) += rev_kernel(k+left)*in(l+k);
                            }
                        }
                    }
                    else if(end + k > in.shape(0))
                    {
                        // convolution of interior
                        if(use_simd)
                        {
                            detail::simd_fma_row(&in(k), in.shape(0)-k, &out(0), rev_kernel(k+left));
                        }
                        else
                        {
                            // out.view(slice(0, in.shape(0)-k)) += rev_kernel(k+left)*in.view(slice(k, in.shape(0)));
                            for(index_t l=0; l<in.shape(0)-k; ++l)
                            {
                                out(l) += rev_kernel(k+left)*in(l+k);
                            }
                        }

                        // right border treatment
                        // invariant: right_padding != no_padding
                        if(right_padding == reflect_padding)
                        {
                            for(index_t l=0; l<k; ++l)
                            {
                                out(in.shape(0)-k+l) += rev_kernel(k+left)*in(in.shape(0)-l-2);
                            }
                        }
                        else if(right_padding == reflect0_padding)
                        {
                            for(index_t l=0; l<k; ++l)
                            {
                                out(in.shape(0)-k+l) += rev_kernel(k+left)*in(in.shape(0)-l-1);
                            }
                        }
                        else if(right_padding == repeat_padding)
                        {
                            for(index_t l=0; l<k; ++l)
                            {
                                out(in.shape(0)-k+l) += rev_kernel(k+left)*in(in.shape(0)-1);
                            }
                        }
                        else if(right_padding == periodic_padding)
                        {
                            for(index_t l=0; l<k; ++l)
                            {
                                out(in.shape(0)-k+l) += rev_kernel(k+left)*in(l);
                            }
                        }
                        // else if(right_padding == zero_padding) pass;
                    }
                    else
                    {
                        // convolution of interior
                        // invariants: left_padding == no_padding || right_padding == no_padding
                        if(use_simd)
                        {
                            // detail::simd_fma_row_symmetric(&in(start+k), &in(start-k), end-start, &out(start), rev_kernel(k+left));
                            detail::simd_fma_row(&in(start+k), end-start, &out(start), rev_kernel(k+left));
                        }
                        else
                        {
                            // out.view(slice(start, end)) += rev_kernel(k+left)*in.view(slice(start+k, end+k));
                            for(index_t l=start; l<end; ++l)
                            {
                                out(l) += rev_kernel(k+left)*in(l+k);
                            }
                        }
                    }
                }
            }
       }


        bool adjust_index_near_border(index_t & i, index_t size,
                                      padding_mode left_padding, padding_mode right_padding) const
        {
            if(i < 0)
            {
                // left border adjustment
                // never reached when left_padding == no_padding
                if(left_padding == zero_padding)
                {
                    return false;
                }
                if(left_padding == reflect_padding)
                {
                    i = -i;
                }
                else if(left_padding == reflect0_padding)
                {
                    i = -i - 1;
                }
                else if(left_padding == periodic_padding)
                {
                    i += size;
                }
                else if(left_padding == repeat_padding)
                {
                    i = 0;
                }
            }
            else if(i >= size)
            {
                // right border adjustment
                // never reached when right_padding == no_padding
                if(right_padding == zero_padding)
                {
                    return false;
                }
                if(right_padding == reflect_padding)
                {
                    i = 2*size - i - 2;
                }
                else if(right_padding == reflect0_padding)
                {
                    i = 2*size - i - 1;
                }
                else if(right_padding == periodic_padding)
                {
                    i -= size;
                }
                else if(right_padding == repeat_padding)
                {
                    i = size - 1;
                }
            }
            return true;
        }

        template <class T1, class T2, class T3>
        void convolve_columns(view_nd<T1, 2> && in, view_nd<T2, 2> && out,
                              kernel_1d<T3> const & kernel, bool use_simd,
                              padding_mode left_padding, padding_mode right_padding) const
        {
#ifdef XVIGRA_USE_SIMD
            use_simd = use_simd &&
                       in.bind(0,0).is_contiguous() && out.bind(0,0).is_contiguous() &&
                       std::is_same<T1, T2>::value && std::is_floating_point<T1>::value;
#else
            use_simd = false;
#endif
            using namespace slicing;
            auto rev_kernel = kernel.view(slice(_,_,-1));
            index_t right = kernel.center(),
                    left  = kernel.size() - right - 1;
            // FIXME: optimize for (a)symmetric kernels
            index_t start = (left_padding == no_padding) ? left : 0;
            index_t end   = (right_padding == no_padding) ? in.shape(0) - right : in.shape(0);
            for(index_t j=start; j<end; ++j)
            {
                if(use_simd)
                {
                    detail::simd_mul_row(&in(j,0), in.shape(1), &out(j,0), rev_kernel(left));
                }
                else
                {
                    // out.bind(0, j) += rev_kernel(left)*in.bind(0,j);
                    for(index_t l=0; l<in.shape(1); ++l)
                    {
                        out(j,l) = rev_kernel(left)*in(j,l);
                    }
                }
                for(index_t k=-left; k<=right; ++k)
                {
                    if(k==0)
                    {
                        continue;
                    }
                    index_t i = j + k;
                    if(!adjust_index_near_border(i, in.shape(0), left_padding, right_padding))
                    {
                        continue; // if zero_padding
                    }

                    if(use_simd)
                    {
                        detail::simd_fma_row(&in(i,0), in.shape(1), &out(j,0), rev_kernel(k+left));
                    }
                    else
                    {
                        // out.bind(0, j) += rev_kernel(k+left)*in.bind(0,i);
                        for(index_t l=0; l<in.shape(1); ++l)
                        {
                            out(j,l) += rev_kernel(k+left)*in(i,l);
                        }
                    }
                }
                // // experimental: use symmetry of the kernel, minimal speed advantage (within measurement noise)
                // for(index_t k=-left; k<0; ++k)
                // {
                //     index_t i1 = j + k,
                //             i2 = j - k;
                //     if(!adjust_index_near_border(i1, in.shape(0), left_padding, right_padding) ||
                //        !adjust_index_near_border(i2, in.shape(0), left_padding, right_padding))
                //     {
                //         continue;
                //     }

                //     // convolution of interior
                //     if(use_simd)
                //     {
                //         detail::simd_fma_row_symmetric(&in(i1,0), &in(i2,0), in.shape(1), &out(j,0), rev_kernel(k+left));
                //     }
                //     else
                //     {
                //         // out.bind(0, j) += rev_kernel(k)*(in.bind(0,i1)+in.bind(0,i2));
                //         for(index_t l=0; l<in.shape(1); ++l)
                //         {
                //             out(j,l) += rev_kernel(k+left)*(in(i1,l) + in(i2,l));
                //         }
                //     }
                // }
            }
        }
    };

    namespace
    {
        separable_convolution_functor  separable_convolution;

        inline void separable_convolution_dummy()
        {
            std::ignore = separable_convolution;
        }
    }

}

#endif // XVIGRA_SEPARABLE_CONVOLUTION_HPP
