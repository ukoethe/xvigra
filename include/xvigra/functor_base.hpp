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

#ifndef XVIGRA_FUNCTOR_BASE_HPP
#define XVIGRA_FUNCTOR_BASE_HPP

#include "global.hpp"
#include "array_nd.hpp"

namespace xvigra
{
    /****************/
    /* functor_base */
    /****************/

    template <class DERIVED>
    struct functor_base
    {
        functor_base()
        {}

        DERIVED const & derived_cast() const
        {
            return static_cast<DERIVED const &>(*this);
        }

        std::string name() const
        {
            return derived_cast().name;
        }

        // FIXME: functor_base should support value_type=tiny_vector

        template <class E1, class E2, class ... ARGS>
        void operator()(E1 && e1, E2 && e2, ARGS ... a) const
        {
            auto && a1 = eval_expr(std::forward<E1>(e1));
            auto && a2 = eval_expr(std::forward<E2>(e2));
            derived_cast().impl(make_view(a1), make_view(a2), std::forward<ARGS>(a)...);
        }

        template <class E1, class E2, class ... ARGS>
        void operator()(dimension_hint dim, E1 && e1, E2 && e2, ARGS ... a) const
        {
            vigra_precondition((index_t)e1.dimension() == dim || (index_t)e1.dimension() == dim+1,
                name() + "(): input dimension contradicts dimension_hint.");

            auto && a1 = eval_expr(std::forward<E1>(e1));
            auto && a2 = eval_expr(std::forward<E2>(e2));

            if((index_t)a1.dimension() == dim)
            {
                derived_cast().impl(make_view(a1), make_view(a2), std::forward<ARGS>(a)...);
            }
            else
            {
                auto && v1 = make_view(a1);
                auto && v2 = make_view(a2);
                for(index_t k=0; k<v1.shape(dim); ++k)
                {
                    derived_cast().impl(v1.bind(dim, k), v2.bind(dim, k), std::forward<ARGS>(a)...);
                }
            }
        }
    };
}

#endif // XVIGRA_FUNCTOR_BASE_HPP
