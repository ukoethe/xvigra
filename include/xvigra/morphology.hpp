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

#ifndef XVIGRA_MORPHOLOGY_HPP
#define XVIGRA_MORPHOLOGY_HPP

#include "global.hpp"
#include "error.hpp"
#include "distance_transform.hpp"

namespace xvigra
{
	namespace detail 
	{

		// this class simplifies the design, but more importantly, it makes sure
		// that the in-place code doesn't get compiled for boolean arrays 
		// (were it would never executed anyway -- see the specializations below)
		template <class DestType, class TmpType>
		struct binary_morphology_impl
		{
		    template <class InArray, class OutArray>
		    static void
		    exec( InArray const & in, OutArray && out, 
		          double radius, bool dilation)
		    {
		        // work on a real-valued temporary array if the squared distances wouldn't fit
		        rebind_container_t<OutArray, TmpType> tmp(out.shape());
		            
		        distance_transform_squared(in, tmp, dilation);
		            
		        // threshold everything less than radius away from the edge
		        double radius2 = radius * radius;
		        int foreground = dilation 
		                            ? 0
		                            : 1,
		            background = dilation 
		                            ? 1
		                            : 0;
		        out = where(tmp > radius2, foreground, background);
		    }
		};

		template <class DestType>
		struct binary_morphology_impl<DestType, DestType>
		{
		    template <class InArray, class OutArray>
		    static void
		    exec( InArray const & in, OutArray && out, 
		          double radius, bool dilation)
		    {
		        distance_transform_squared(in, out, dilation);
		            
		        // threshold everything less than radius away from the edge
		        double radius2 = radius * radius;
		        int foreground = dilation 
		                            ? 0
		                            : 1,
		            background = dilation 
		                            ? 1
		                            : 0;
		        out = where(out > radius2, foreground, background);
		    }
		};

		template <>
		struct binary_morphology_impl<bool, bool>
		{
		    template <class InArray, class OutArray>
		    static void
		    exec( InArray const &, OutArray &&, 
		          double, bool)
		    {
		        vigra_fail("binary_morphology(): Internal error (this function should never be called).");
		    }
		};

	} // namespace detail

	/** \addtogroup MultiArrayMorphology Morphological operators for multi-dimensional arrays.

	    These functions perform morphological operations on an arbitrary
	    dimensional array that is specified by iterators (compatible to \ref MultiIteratorPage)
	    and shape objects. It can therefore be applied to a wide range of data structures
	    (\ref vigra::MultiArrayView, \ref vigra::MultiArray etc.).
	*/
	//@{

    /******************/
    /* binary_erosion */
    /******************/

	/** \brief Binary erosion on multi-dimensional arrays.

	    This function applies a flat circular erosion operator with a given radius. The
	    operation is isotropic. The input is interpreted as a binary multi-dimensional 
	    array where non-zero pixels represent foreground and zero pixels represent 
	    background. In the output, foreground is always represented by ones 
	    (i.e. NumericTrais<typename DestAccessor::value_type>::one()).
	    
	    This function may work in-place, which means that <tt>siter == diter</tt> is allowed.
	    A temporary internal array is only allocated if working on the destination
	    array directly would cause overflow errors (that is if
	    <tt> NumericTraits<typename DestAccessor::value_type>::max() < squaredNorm(shape)</tt>, 
	    i.e. the squared length of the image diagonal doesn't fit into the destination type).
	           
	    <b> Declarations:</b>

	    pass arbitrary-dimensional array views:
	    \code
	    namespace vigra {
	        template <unsigned int N, class T1, class S1,
	                                  class T2, class S2>
	        void
	        multiBinaryErosion(MultiArrayView<N, T1, S1> const & source,
	                           MultiArrayView<N, T2, S2> dest, 
	                           double radius);
	    }
	    \endcode

	    \deprecatedAPI{multiBinaryErosion}
	    pass \ref MultiIteratorPage "MultiIterators" and \ref DataAccessors :
	    \code
	    namespace vigra {
	        template <class SrcIterator, class SrcShape, class SrcAccessor,
	                  class DestIterator, class DestAccessor>
	        void
	        multiBinaryErosion(SrcIterator siter, SrcShape const & shape, SrcAccessor src,
	                                    DestIterator diter, DestAccessor dest, int radius);

	    }
	    \endcode
	    use argument objects in conjunction with \ref ArgumentObjectFactories :
	    \code
	    namespace vigra {
	        template <class SrcIterator, class SrcShape, class SrcAccessor,
	                  class DestIterator, class DestAccessor>
	        void
	        multiBinaryErosion(triple<SrcIterator, SrcShape, SrcAccessor> const & source,
	                                    pair<DestIterator, DestAccessor> const & dest, 
	                                    int radius);

	    }
	    \endcode
	    \deprecatedEnd

	    <b> Usage:</b>

	    <b>\#include</b> \<vigra/multi_morphology.hxx\><br/>
	    Namespace: vigra

	    \code
	    Shape3 shape(width, height, depth);
	    MultiArray<3, unsigned char> source(shape);
	    MultiArray<3, unsigned char> dest(shape);
	    ...

	    // perform isotropic binary erosion
	    multiBinaryErosion(source, dest, 3);
	    \endcode

	    \see vigra::discErosion(), vigra::multiGrayscaleErosion()
	*/
	// doxygen_overloaded_function(template <...> void multiBinaryErosion)

	template <class InArray, class OutArray,
	          VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
	void
	binary_erosion(InArray const & in, OutArray && out, double radius)
	{
	    using dest_type = typename std::decay_t<OutArray>::value_type;
	    
	    double dmax = norm_sq(in.shape()) + 1.0;

	    // Get the distance squared transform of the image
	    if(dmax > std::numeric_limits<dest_type>::max())
	    {
	        detail::binary_morphology_impl<dest_type, std::int64_t>::exec(in, out, radius, false);
	    }
	    else    // work directly on the destination array
	    {
	        detail::binary_morphology_impl<dest_type, dest_type>::exec(in, out, radius, false);
	    }
	}


    /*******************/
    /* binary_dilation */
    /*******************/

	/** \brief Binary dilation on multi-dimensional arrays.

	    This function applies a flat circular dilation operator with a given radius. The
	    operation is isotropic. The input is interpreted as a binary multi-dimensional 
	    array where non-zero pixels represent foreground and zero pixels represent 
	    background. In the output, foreground is always represented by ones 
	    (i.e. NumericTrais<typename DestAccessor::value_type>::one()).
	    
	    This function may work in-place, which means that <tt>siter == diter</tt> is allowed.
	    A temporary internal array is only allocated if working on the destination
	    array directly would cause overflow errors (that is if
	    <tt> NumericTraits<typename DestAccessor::value_type>::max() < squaredNorm(shape)</tt>, 
	    i.e. the squared length of the image diagonal doesn't fit into the destination type).
	           
	    <b> Declarations:</b>

	    pass arbitrary-dimensional array views:
	    \code
	    namespace vigra {
	        template <unsigned int N, class T1, class S1,
	                                  class T2, class S2>
	        void 
	        multiBinaryDilation(MultiArrayView<N, T1, S1> const & source,
	                            MultiArrayView<N, T2, S2> dest,
	                            double radius);
	    }
	    \endcode

	    \deprecatedAPI{multiBinaryDilation}
	    pass \ref MultiIteratorPage "MultiIterators" and \ref DataAccessors :
	    \code
	    namespace vigra {
	        template <class SrcIterator, class SrcShape, class SrcAccessor,
	                  class DestIterator, class DestAccessor>
	        void
	        multiBinaryDilation(SrcIterator siter, SrcShape const & shape, SrcAccessor src,
	                                    DestIterator diter, DestAccessor dest, int radius);

	    }
	    \endcode
	    use argument objects in conjunction with \ref ArgumentObjectFactories :
	    \code
	    namespace vigra {
	        template <class SrcIterator, class SrcShape, class SrcAccessor,
	                  class DestIterator, class DestAccessor>
	        void
	        multiBinaryDilation(triple<SrcIterator, SrcShape, SrcAccessor> const & source,
	                                    pair<DestIterator, DestAccessor> const & dest, 
	                                    int radius);

	    }
	    \endcode
	    \deprecatedEnd

	    <b> Usage:</b>

	    <b>\#include</b> \<vigra/multi_morphology.hxx\><br/>
	    Namespace: vigra

	    \code
	    Shape3 shape(width, height, depth);
	    MultiArray<3, unsigned char> source(shape);
	    MultiArray<3, unsigned char> dest(shape);
	    ...

	    // perform isotropic binary erosion
	    multiBinaryDilation(source, dest, 3);
	    \endcode

	    \see vigra::discDilation(), vigra::multiGrayscaleDilation()
	*/
	// doxygen_overloaded_function(template <...> void multiBinaryDilation)

	template <class InArray, class OutArray,
	          VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
	void
	binary_dilation(InArray const & in, OutArray && out, double radius)
	{
	    using dest_type = typename std::decay_t<OutArray>::value_type;
	    
	    double dmax = norm_sq(in.shape()) + 1.0;

	    // Get the distance squared transform of the image
	    if(dmax > std::numeric_limits<dest_type>::max())
	    {
	        detail::binary_morphology_impl<dest_type, std::int64_t>::exec(in, out, radius, true);
	    }
	    else    // work directly on the destination array
	    {
	        detail::binary_morphology_impl<dest_type, dest_type>::exec(in, out, radius, true);
	    }
	}

    /******************/
    /* binary_opening */
    /******************/

	template <class InArray, class OutArray,
	          VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
	void
	binary_opening(InArray const & in, OutArray && out, double radius)
	{
	    using dest_type = typename std::decay_t<OutArray>::value_type;
	    
	    double dmax = norm_sq(in.shape()) + 1.0;

	    // Get the distance squared transform of the image
	    if(dmax > std::numeric_limits<dest_type>::max())
	    {
	        detail::binary_morphology_impl<dest_type, std::int64_t>::exec(in, out, radius, false);
	        detail::binary_morphology_impl<dest_type, std::int64_t>::exec(out, out, radius, true);
	    }
	    else    // work directly on the destination array
	    {
	        detail::binary_morphology_impl<dest_type, dest_type>::exec(in, out, radius, false);
	        detail::binary_morphology_impl<dest_type, dest_type>::exec(out, out, radius, true);
	    }
	}

    /******************/
    /* binary_closing */
    /******************/

	template <class InArray, class OutArray,
	          VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
	void
	binary_closing(InArray const & in, OutArray && out, double radius)
	{
	    using dest_type = typename std::decay_t<OutArray>::value_type;
	    
	    double dmax = norm_sq(in.shape()) + 1.0;

	    // Get the distance squared transform of the image
	    if(dmax > std::numeric_limits<dest_type>::max())
	    {
	        detail::binary_morphology_impl<dest_type, std::int64_t>::exec(in, out, radius, true);
	        detail::binary_morphology_impl<dest_type, std::int64_t>::exec(out, out, radius, false);
	    }
	    else    // work directly on the destination array
	    {
	        detail::binary_morphology_impl<dest_type, dest_type>::exec(in, out, radius, true);
	        detail::binary_morphology_impl<dest_type, dest_type>::exec(out, out, radius, false);
	    }
	}

    /*********************/
    /* grayscale_erosion */
    /*********************/

	/** \brief Parabolic grayscale erosion on multi-dimensional arrays.

	    This function applies a parabolic erosion operator with a given spread (sigma) on
	    a grayscale array. The operation is isotropic.
	    The input is a grayscale multi-dimensional array.
	    
	    This function may work in-place, which means that <tt>siter == diter</tt> is allowed.
	    A full-sized internal array is only allocated if working on the destination
	    array directly would cause overflow errors (i.e. if
	    <tt> typeid(typename DestAccessor::value_type) < N * M*M</tt>, where M is the
	    size of the largest dimension of the array.
	           
	    <b> Declarations:</b>

	    pass arbitrary-dimensional array views:
	    \code
	    namespace vigra {
	        template <unsigned int N, class T1, class S1,
	                                  class T2, class S2>
	        void
	        multiGrayscaleErosion(MultiArrayView<N, T1, S1> const & source,
	                              MultiArrayView<N, T2, S2> dest, 
	                              double sigma);
	    }
	    \endcode

	    \deprecatedAPI{multiGrayscaleErosion}
	    pass \ref MultiIteratorPage "MultiIterators" and \ref DataAccessors :
	    \code
	    namespace vigra {
	        template <class SrcIterator, class SrcShape, class SrcAccessor,
	                  class DestIterator, class DestAccessor>
	        void
	        multiGrayscaleErosion(SrcIterator siter, SrcShape const & shape, SrcAccessor src,
	                                    DestIterator diter, DestAccessor dest, double sigma);

	    }
	    \endcode
	    use argument objects in conjunction with \ref ArgumentObjectFactories :
	    \code
	    namespace vigra {
	        template <class SrcIterator, class SrcShape, class SrcAccessor,
	                  class DestIterator, class DestAccessor>
	        void
	        multiGrayscaleErosion(triple<SrcIterator, SrcShape, SrcAccessor> const & source,
	                                    pair<DestIterator, DestAccessor> const & dest, 
	                                    double sigma);

	    }
	    \endcode
	    \deprecatedEnd

	    <b> Usage:</b>

	    <b>\#include</b> \<vigra/multi_morphology.hxx\><br/>
	    Namespace: vigra

	    \code
	    Shape3 shape(width, height, depth);
	    MultiArray<3, unsigned char> source(shape);
	    MultiArray<3, unsigned char> dest(shape);
	    ...

	    // perform isotropic grayscale erosion
	    multiGrayscaleErosion(source, dest, 3.0);
	    \endcode

	    \see vigra::discErosion(), vigra::multiBinaryErosion()
	*/
	// doxygen_overloaded_function(template <...> void multiGrayscaleErosion)

	template <class InArray, class OutArray,
	          VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
	void
	grayscale_erosion(InArray const & in, OutArray && out, double sigma)
	{
		std::vector<double> pixel_pitch(in.shape().size(), 1.0 / sigma);
		detail::distance_transform_impl(in, out, pixel_pitch);
	}

    /**********************/
    /* grayscale_dilation */
    /**********************/

	/** \brief Parabolic grayscale dilation on multi-dimensional arrays.

	    This function applies a parabolic dilation operator with a given spread (sigma) on
	    a grayscale array. The operation is isotropic.
	    The input is a grayscale multi-dimensional array.
	    
	    This function may work in-place, which means that <tt>siter == diter</tt> is allowed.
	    A full-sized internal array is only allocated if working on the destination
	    array directly would cause overflow errors (i.e. if
	    <tt> typeid(typename DestAccessor::value_type) < N * M*M</tt>, where M is the
	    size of the largest dimension of the array.
	           
	    <b> Declarations:</b>

	    pass arbitrary-dimensional array views:
	    \code
	    namespace vigra {
	        template <unsigned int N, class T1, class S1,
	                                  class T2, class S2>
	        void
	        multiGrayscaleDilation(MultiArrayView<N, T1, S1> const & source,
	                               MultiArrayView<N, T2, S2> dest,
	                               double sigma);
	    }
	    \endcode

	    \deprecatedAPI{multiGrayscaleDilation}
	    pass \ref MultiIteratorPage "MultiIterators" and \ref DataAccessors :
	    \code
	    namespace vigra {
	        template <class SrcIterator, class SrcShape, class SrcAccessor,
	                  class DestIterator, class DestAccessor>
	        void
	        multiGrayscaleDilation(SrcIterator siter, SrcShape const & shape, SrcAccessor src,
	                                    DestIterator diter, DestAccessor dest, double sigma);

	    }
	    \endcode
	    use argument objects in conjunction with \ref ArgumentObjectFactories :
	    \code
	    namespace vigra {
	        template <class SrcIterator, class SrcShape, class SrcAccessor,
	                  class DestIterator, class DestAccessor>
	        void
	        multiGrayscaleDilation(triple<SrcIterator, SrcShape, SrcAccessor> const & source,
	                                    pair<DestIterator, DestAccessor> const & dest, 
	                                    double sigma);

	    }
	    \endcode
	    \deprecatedEnd

	    <b> Usage:</b>

	    <b>\#include</b> \<vigra/multi_morphology.hxx\><br/>
	    Namespace: vigra

	    \code
	    Shape3 shape(width, height, depth);
	    MultiArray<3, unsigned char> source(shape);
	    MultiArray<3, unsigned char> dest(shape);
	    ...

	    // perform isotropic grayscale erosion
	    multiGrayscaleDilation(source, dest, 3.0);
	    \endcode

	    \see vigra::discDilation(), vigra::multiBinaryDilation()
	*/
	// doxygen_overloaded_function(template <...> void multiGrayscaleDilation)

	template <class InArray, class OutArray,
	          VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
	void
	grayscale_dilation(InArray const & in, OutArray && out, double sigma)
	{
		std::vector<double> pixel_pitch(in.shape().size(), 1.0 / sigma);
		detail::distance_transform_impl(in, out, pixel_pitch, true);
	}

    /*********************/
    /* grayscale_opening */
    /*********************/

	template <class InArray, class OutArray,
	          VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
	void
	grayscale_opening(InArray const & in, OutArray && out, double sigma)
	{
		std::vector<double> pixel_pitch(in.shape().size(), 1.0 / sigma);
		detail::distance_transform_impl(in, out, pixel_pitch, false);
		detail::distance_transform_impl(out, out, pixel_pitch, true);
	}

    /*********************/
    /* grayscale_closing */
    /*********************/

	template <class InArray, class OutArray,
	          VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
	void
	grayscale_closing(InArray const & in, OutArray && out, double sigma)
	{
		std::vector<double> pixel_pitch(in.shape().size(), 1.0 / sigma);
		detail::distance_transform_impl(in, out, pixel_pitch, true);
		detail::distance_transform_impl(out, out, pixel_pitch, false);
	}
//@}

} // namespace xvigra

#endif // XVIGRA_MORPHOLOGY_HPP
