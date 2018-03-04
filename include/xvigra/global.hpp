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

#ifndef XVIGRA_GLOBAL_HPP
#define XVIGRA_GLOBAL_HPP

#include <cstddef>
#include <vector>
#include <array>
#include <xtensor/xtensor_forward.hpp>

namespace xvigra
{
    /***********/
    /* index_t */
    /***********/

	using index_t = std::ptrdiff_t;

    /********************/
    /* rebind_container */
    /********************/

    template <class C, class T>
    struct rebind_container;

    template <class C, class T>
    using rebind_container_t = typename rebind_container<C, T>::type;

    template <class T, class A, class NT>
    struct rebind_container<std::vector<T, A>, NT>
    {
        using ALLOC = typename A::template rebind<NT>::other;
        using type = std::vector<NT, ALLOC>;
    };

    template <class T, std::size_t N, class NT>
    struct rebind_container<std::array<T, N>, NT>
    {
        using type = std::array<NT, N>;
    };

    template <class T, xt::layout_type L, class NT>
    struct rebind_container<xt::xarray<T, L>, NT>
    {
        using type = xt::xarray<NT, L>;
    };

    template <class T, std::size_t N, xt::layout_type L, class NT>
    struct rebind_container<xt::xtensor<T, N, L>, NT>
    {
        using type = xt::xtensor<NT, N, L>;
    };
}

#endif // XVIGRA_GLOBAL_HPP
