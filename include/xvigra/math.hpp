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
#include "concepts.hpp"

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

    template <class T,
              bool IS_ARITHMETIC=std::is_arithmetic<T>::value,
              bool IS_INTEGRAL=std::is_integral<T>::value,
              bool IS_CONTAINER=container_concept<T>::value>
    struct default_tolerance;

    template <class T>
    struct default_tolerance<T, true, false, false> // T is floating point
    {
        static constexpr double value = 2.0*std::numeric_limits<T>::epsilon();
    };

    template <class T>
    struct default_tolerance<T, true, true, false> // T is integral
    {
        static constexpr double value = 0.0;
    };

    template <class T>
    struct default_tolerance<T, false, false, true> // T is container
    {
        static constexpr double value = default_tolerance<typename T::value_type>::value;
    };

    template <class T>
    inline bool
    is_close(T const & a, T const & b,
             double rtol = default_tolerance<T>::value,
             double atol = default_tolerance<T>::value,
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
        return d <= atol || d <= rtol * double(std::max<internal_type>(math::abs(a), math::abs(b)));
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

    /*******/
    /* min */
    /*******/

        /** \brief A proper minimum function.

            The <tt>std::min</tt> template matches everything -- this is way too
            greedy to be useful. xvigra implements the basic <tt>min</tt> function
            only for arithmetic types and provides explicit overloads for everything
            else. Moreover, xvigra's <tt>min</tt> function also computes the minimum
            between two different types, as long as they have a <tt>std::common_type</tt>.

            <b>\#include</b> \<xvigra/math.hpp\><br>
            Namespace: xvigra
        */
    template <class T1, class T2,
              VIGRA_REQUIRE<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value> >
    inline std::common_type_t<T1, T2>
    min(T1 const & t1, T2 const & t2)
    {
        return std::min<std::common_type_t<T1, T2>>(t1, t2);
    }

    template <class T,
              VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
    inline T const &
    min(T const & t1, T const & t2)
    {
        return std::min(t1, t2);
    }

    /*******/
    /* max */
    /*******/

        /** \brief A proper maximum function.

            The <tt>std::max</tt> template matches everything -- this is way too
            greedy to be useful. xvigra implements the basic <tt>max</tt> function
            only for arithmetic types and provides explicit overloads for everything
            else. Moreover, xvigra's <tt>max</tt> function also computes the maximum
            between two different types, as long as they have a <tt>std::common_type</tt>.

            <b>\#include</b> \<xvigra/math.hpp\><br>
            Namespace: vigra
        */
    template <class T1, class T2,
              VIGRA_REQUIRE<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value> >
    inline std::common_type_t<T1, T2>
    max(T1 const & t1, T2 const & t2)
    {
        return std::max<std::common_type_t<T1, T2>>(t1, t2);
    }

    template <class T,
              VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
    inline T const &
    max(T const & t1, T const & t2)
    {
        return std::max(t1, t2);
    }

} // namespace xvigra

#endif // XVIGRA_MATH_HPP
