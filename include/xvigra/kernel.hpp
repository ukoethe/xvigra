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

#ifndef XVIGRA_KERNEL_HPP
#define XVIGRA_KERNEL_HPP

#include "array_nd.hpp"
#include "gaussian.hpp"

namespace xvigra
{
    template <class T>
    class kernel_1d
    : public array_nd<T, 1>
    {
      public:
        using base_type = array_nd<T, 1>;

        explicit kernel_1d(index_t size)
        : kernel_1d(size, size/2)
        {}

        kernel_1d(index_t size, index_t center)
        : base_type(shape_t<1>{size})
        , center_(center)
        {
            vigra_precondition(center >= 0 && center < size,
                "kernel_1d(): center must be inside the kernel.");
        }

        kernel_1d(kernel_1d const & other)
        : base_type(other)
        , center_(other.center_)
        {}

        using base_type::operator=;

        index_t center() const
        {
            return center_;
        }

        index_t center_;
    };

    template <class T=double>
    inline kernel_1d<T>
    averaging_kernel_1d(index_t radius)
    {
        kernel_1d<T> res(2*radius+1, radius);
        res = static_cast<T>(1.0 / res.size());
        return res;
    }

    template <class T=double>
    inline kernel_1d<T>
    gaussian_kernel_1d(double sigma, index_t radius)
    {
        kernel_1d<T> res(2*radius+1, radius);

        gaussian<T> gauss(sigma);
        T sum = 0;
        for(index_t k=-radius; k<=radius; ++k)
        {
            T g = gauss(k);
            res(k+radius) = g;
            sum += g;
        }
        res *= T(1)/sum;
        return res;
    }

    template <class T=double>
    inline kernel_1d<T>
    gaussian_kernel_1d(double sigma)
    {
        return gaussian_kernel_1d<T>(sigma, (index_t)(3.0 * sigma + 0.5));
    }

    template <class T=double>
    inline kernel_1d<T>
    gaussian_derivative_kernel_1d(double sigma, index_t order, index_t radius)
    {
        kernel_1d<T> res(2*radius+1, radius);

        gaussian<T> gauss(sigma, order);
        T sum = 0;
        for(index_t k=-radius; k<=radius; ++k)
        {
            T g = gauss(k);
            res(k+radius) = g;
            sum += g;
        }
        if(order > 0)
        {
            res -= sum; // DC correction
            sum = 0;
            for(index_t k=-radius; k<=radius; ++k)
            {
                sum += res(k+radius) * math::pow(-k, order);
            }
            for(index_t i = 2; i <= order; ++i)
            {
                sum /= i;
            }
        }
        res *= T(1) / sum;
        return res;
    }

    template <class T=double>
    inline kernel_1d<T>
    gaussian_derivative_kernel_1d(double sigma, index_t order)
    {
        return gaussian_derivative_kernel_1d<T>(sigma, order, (index_t)((3.0 + 0.5*order) * sigma + 0.5));
    }
}

#endif // XVIGRA_KERNEL_HPP
