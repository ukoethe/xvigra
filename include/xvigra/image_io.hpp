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

#ifndef XVIGRA_IMAGE_IO_HPP
#define XVIGRA_IMAGE_IO_HPP

#include <OpenImageIO/imageio.h>
#include <xtensor/xmath.hpp>
#include <xtensor/xeval.hpp>
#include "array_nd.hpp"

namespace xvigra
{
    /**
     * Load an image from file at filename.
     * Storage format is deduced from file ending.
     *
     * @param filename The path of the file to load
     *
     * @return array_nd with image contents. The shape of the xarray
     *         is ``HEIGHT x WIDTH x CHANNELS`` of the loaded image, where
     *         ``CHANNELS`` are ordered in standard ``R G B (A)``.
     */
    template <class T = unsigned char>
    array_nd<T> read_image(std::string filename)
    {
        auto close_file  =  [](OIIO::ImageInput * file)
                            {
                                file->close();
                                OIIO::ImageInput::destroy(file);
                            };

        std::unique_ptr<OIIO::ImageInput, decltype(close_file)> in(OIIO::ImageInput::open(filename), close_file);
        vigra_precondition(!!in,
            "read_image(): Error reading image '" + filename + "'.");

        const OIIO::ImageSpec& spec = in->spec();

        shape_t<> shape{spec.height, spec.width};
        if(spec.nchannels > 1)
        {
            shape = shape.push_back(spec.nchannels);
        }

        array_nd<T> image(shape);

        in->read_image(OIIO::BaseTypeFromC<T>::value, image.raw_data());

        return image;
    }

        /** \brief Pass options to write_image().
        */
    struct write_image_options
    {
            /** \brief Initialize options to default values.
            */
        write_image_options()
        : spec(0,0,0)
        , autoconvert(true)
        {
            spec.attribute("CompressionQuality", 90);
        }

            /** \brief Forward an attribute to an OpenImageIO ImageSpec.

                See the documentation of OIIO::ImageSpec::attribute() for a list
                of supported attributes.

                Default: "CompressionQuality" = 90
            */
        template <class T>
        write_image_options & attribute(OIIO::string_view name, T const & v)
        {
            spec.attribute(name, v);
            return *this;
        }

        OIIO::ImageSpec spec;
        bool autoconvert;
    };

    /**
     * Save image to disk.
     * The desired image format is deduced from ``filename``.
     * Supported formats are those supported by OpenImageIO.
     * Most common formats are supported (jpg, png, gif, bmp, tiff).
     * The shape of the array must be ``HEIGHT x WIDTH`` or ``HEIGHT x WIDTH x CHANNELS``.
     *
     * @param filename The path to the desired file
     * @param data Image data
     * @param options Pass a write_image_options object to fine-tune image export
     */
    template <class E>
    void write_image(std::string filename, const xt::xexpression<E>& data,
                     write_image_options const & options = write_image_options())
    {
        E const & e = data.derived_cast();
        using value_type = typename std::decay_t<E>::value_type;

        auto shape = e.shape();
        vigra_precondition(shape.size() == 2 || shape.size() == 3,
            "write_image(): data must have 2 or 3 dimensions (channels must be last).");

        auto close_file  =  [](OIIO::ImageOutput * file)
                            {
                                file->close();
                                OIIO::ImageOutput::destroy(file);
                            };
        std::unique_ptr<OIIO::ImageOutput, decltype(close_file)> out(OIIO::ImageOutput::create(filename), close_file);
        vigra_precondition(!!out,
            "write_image(): Error opening file '" + filename + "' to write image.");

        OIIO::ImageSpec spec = options.spec;

        spec.width     = static_cast<int>(shape[1]);
        spec.height    = static_cast<int>(shape[0]);
        spec.nchannels = (shape.size() == 2)
                           ? 1
                           : static_cast<int>(shape[2]);
        spec.format    = OIIO::BaseTypeFromC<value_type>::value;

        out->open(filename, spec);

        auto && ex = xt::eval(e);
        if(out->spec().format != OIIO::BaseTypeFromC<value_type>::value)
        {
            // OpenImageIO changed the target type because the file format doesn't support value_type.
            // It will do automatic conversion, but the data should be in the range 0...1
            // for good results.
            auto mM = minmax(ex)();

            if(mM[0] != mM[1])
            {
                using real_t = real_promote_type_t<value_type>;
                auto && normalized = xt::eval((real_t(1.0) / (mM[1] - mM[0])) * (ex - mM[0]));
                out->write_image(OIIO::BaseTypeFromC<real_t>::value, normalized.raw_data());
            }
            else
            {
                out->write_image(OIIO::BaseTypeFromC<value_type>::value, ex.raw_data());
            }
        }
        else
        {
            out->write_image(OIIO::BaseTypeFromC<value_type>::value, ex.raw_data());
        }
    }
}

#endif // XVIGRA_IMAGE_IO_HPP
