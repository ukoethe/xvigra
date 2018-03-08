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

#ifndef XVIGRA_MATH_HPP
#define XVIGRA_MATH_HPP

#include <cmath>
#include <limits>
#include <complex>
#include <xtensor/xnorm.hpp> // FIXME: includes lots of things
#include <xtensor/xmath.hpp> // FIXME: includes lots of things
#include "global.hpp"

namespace xvigra
{
    /**********************************/
    /* namespace for <cmath> contents */
	/**********************************/

    namespace math = xt::math;
	
    /*********************/
    /* numeric_constants */
	/*********************/

	using xt::numeric_constants;

    /******/
    /* sq */
	/******/

    template <class T>
    inline T sq(T t)
    {
        return t*t;
    }

    /************/
    /* is_close */
	/************/

    template <class T>
    inline bool 
    is_close(T const & a, T const & b, 
              double rtol = 2.0*std::numeric_limits<double>::epsilon(),
              double atol = 2.0*std::numeric_limits<double>::epsilon(),
              bool equal_nan = false)
    {
        using internal_type = promote_type_t<T, double>;
        if(math::isnan(a) && math::isnan(b))
        {
            return equal_nan;
        }
        if(math::isinf(a) && math::isinf(b))
        {
            // check for both infinity signs equal
            return a == b;
        }
        auto d = math::abs(internal_type(a) - internal_type(b));
        return d <= atol || d <= rtol * double(std::max(math::abs(a), math::abs(b)));
    }

    /*********/
    /* norms */
	/*********/

    using xt::norm_lp;
    using xt::norm_lp_to_p;
    using xt::norm_l0;
    using xt::norm_l1;
    using xt::norm_l2;
    using xt::norm_linf;
    using xt::norm_sq;

    template <class T, class A>
    inline auto norm_sq(std::vector<T, A> const & v)
    {
    	using result_type = squared_norm_type_t<std::vector<T, A>>;
    	result_type res = result_type();
    	for(auto u: v)
    	{
    		res += norm_sq(u);
    	}
    	return res;
    }

    template <class T, class A>
    inline auto norm_linf(std::vector<T, A> const & v)
    {
    	using result_type = decltype(norm_linf(v[0]));
    	result_type res = result_type();
    	for(auto u: v)
    	{
    		res = std::max(res, norm_linf(u));
    	}
    	return res;
    }

    template <class T, std::size_t N>
    inline auto norm_sq(std::array<T, N> const & v)
    {
    	using result_type = squared_norm_type_t<std::array<T, N>>;
    	result_type res = result_type();
    	for(auto u: v)
    	{
    		res += norm_sq(u);
    	}
    	return res;
    }

    template <class T, std::size_t N>
    inline auto norm_linf(std::array<T, N> const & v)
    {
    	using result_type = decltype(norm_linf(v[0]));
    	result_type res = result_type();
    	for(auto u: v)
    	{
    		res = std::max(res, norm_linf(u));
    	}
   		return res;
    }

} // namespace xvigra

#endif // XVIGRA_MATH_HPP
