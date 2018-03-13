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

#pragma once

#ifndef XVIGRA_GAUSSIAN_HPP
#define XVIGRA_GAUSSIAN_HPP

#include <cmath>
#include <vector>
#include "global.hpp"
#include "error.hpp"
#include "math.hpp"

namespace xvigra
{

        /** The Gaussian function and its derivatives.

            Implemented as a unary functor. Since it supports the <tt>radius()</tt> function
            it can also be used as a kernel in \ref resamplingConvolveImage().

            <b>\#include</b> \<xvigra/gaussian.hpp\><br>
            Namespace: xvigra

            \ingroup MathFunctions
        */
    template <class T = double>
    class Gaussian
    {
      public:
        using value_type = T;
        using argument_type = T;
        using result_type = T;

            /** Create functor for the given standard deviation <tt>sigma</tt> and
                derivative order <i>n</i>. The functor then realizes the function

                \f[ f_{\sigma,n}(x)=\frac{\partial^n}{\partial x^n}
                     \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{x^2}{2\sigma^2}}
                \f]

                Precondition:
                \code
                sigma > 0.0
                \endcode
            */
        explicit Gaussian(T sigma = 1.0, unsigned int derivative_order = 0)
        : sigma_(sigma),
          sigma2_(T(-0.5 / sigma / sigma)),
          norm_(0.0),
          order_(derivative_order),
          hermite_polynomial_(derivative_order / 2 + 1)
        {
            vigra_precondition(sigma_ > 0.0,
                "Gaussian::Gaussian(): sigma > 0 required.");
            switch(order_)
            {
                case 1:
                case 2:
                    norm_ = T(-1.0 / (std::sqrt(2.0 * M_PI) * sq(sigma) * sigma));
                    break;
                case 3:
                    norm_ = T(1.0 / (std::sqrt(2.0 * M_PI) * sq(sigma) * sq(sigma) * sigma));
                    break;
                default:
                    norm_ = T(1.0 / std::sqrt(2.0 * M_PI) / sigma);
            }
            calculate_hermite_polynomial();
        }

            /** Function (functor) call.
            */
        result_type operator()(argument_type x) const;

            /** Get the standard deviation of the Gaussian.
            */
        value_type sigma() const
        {
            return sigma_;
        }

            /** Get the derivative order of the Gaussian.
            */
        unsigned int derivative_order() const
        {
            return order_;
        }

            /** Get the trauncated filter radius for a discrete approximation of the Gaussian.
                The radius is given as a multiple of the Gaussian's standard deviation
                (default: <tt>sigma * (3 + 1/2 * derivative_order()</tt> -- the second term
                accounts for the fact that the derivatives of the Gaussian become wider
                with increasing order). The result is rounded to the next higher integer.
            */
        double radius(double sigma_multiple = 3.0) const
        {
            return std::ceil(sigma_ * (sigma_multiple + 0.5 * derivative_order()));
        }

      private:
        void calculate_hermite_polynomial();
        T horner(T x) const;

        T sigma_, sigma2_, norm_;
        unsigned int order_;
        std::vector<T> hermite_polynomial_;
    };

    template <class T>
    typename Gaussian<T>::result_type
    Gaussian<T>::operator()(argument_type x) const
    {
        static constexpr bool need_cast = std::is_arithmetic<result_type>::value;

        T x2 = x * x;
        T g  = norm_ * std::exp(x2 * sigma2_);
        switch(order_)
        {
            case 0:
                return conditional_cast<need_cast, result_type>(g);
            case 1:
                return conditional_cast<need_cast, result_type>(x * g);
            case 2:
                return conditional_cast<need_cast, result_type>((1.0 - sq(x / sigma_)) * g);
            case 3:
                return conditional_cast<need_cast, result_type>((3.0 - sq(x / sigma_)) * x * g);
            default:
                return order_ % 2 == 0 ?
                           conditional_cast<need_cast, result_type>(g * horner(x2))
                         : conditional_cast<need_cast, result_type>(x * g * horner(x2));
        }
    }

    template <class T>
    T Gaussian<T>::horner(T x) const
    {
        int i = order_ / 2;
        T res = hermite_polynomial_[i];
        for(--i; i >= 0; --i)
            res = x * res + hermite_polynomial_[i];
        return res;
    }

    template <class T>
    void Gaussian<T>::calculate_hermite_polynomial()
    {
        if(order_ == 0)
        {
            hermite_polynomial_[0] = 1.0;
        }
        else if(order_ == 1)
        {
            hermite_polynomial_[0] = T(-1.0 / sigma_ / sigma_);
        }
        else
        {
            // calculate Hermite polynomial for requested derivative
            // recursively according to
            //     (0)
            //    h   (x) = 1
            //
            //     (1)
            //    h   (x) = -x / s^2
            //
            //     (n+1)                        (n)           (n-1)
            //    h     (x) = -1 / s^2 * [ x * h   (x) + n * h     (x) ]
            //
            T s2 = T(-1.0 / sigma_ / sigma_);
            std::vector<T> hn(3*order_+3, 0.0);
            typename std::vector<T>::pointer  hn0 = hn.data(),
                                              hn1 = hn0 + order_+1,
                                              hn2 = hn1 + order_+1,
                                              ht;
            hn2[0] = 1.0;
            hn1[1] = s2;
            for(unsigned int i = 2; i <= order_; ++i)
            {
                hn0[0] = s2 * (i-1) * hn2[0];
                for(unsigned int j = 1; j <= i; ++j)
                    hn0[j] = s2 * (hn1[j-1] + (i-1) * hn2[j]);
                ht = hn2;
                hn2 = hn1;
                hn1 = hn0;
                hn0 = ht;
            }
            // keep only non-zero coefficients of the polynomial
            for(unsigned int i = 0; i < hermite_polynomial_.size(); ++i)
            {
                hermite_polynomial_[i] = order_ % 2 == 0 ?
                                             hn1[2*i]
                                           : hn1[2*i+1];
            }
        }
    }
} // namespace xvigra


#endif /* XVIGRA_GAUSSIAN_HPP */
