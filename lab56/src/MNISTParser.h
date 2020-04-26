/*
 *  Copyright 2014 Henry Tan
 *  Modified 2018 by Aadyot Bhatngar
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http ://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <assert.h>


//
// C++ MNIST dataset parser
//
// Specification can be found in http://yann.lecun.com/exdb/mnist/
//
class MNISTDataset final
{
public:
    MNISTDataset()
        : m_count(0),
        m_width(0),
        m_height(0),
        m_imageSize(0),
        m_buffer(nullptr),
        m_imageBuffer(nullptr),
        m_categoryBuffer(nullptr)
    {
    }

    ~MNISTDataset()
    {
        if (m_buffer) free(m_buffer);
        if (m_categoryBuffer) free(m_categoryBuffer);
    }

    void Print()
    {
        for (int n = 0; n < m_count; ++n)
        {
            const float* imageBuffer = &m_imageBuffer[n * m_imageSize];
            for (int j = 0; j < m_height; ++j)
            {
                for (int i = 0; i < m_width; ++i)
                {
                    printf("%3d ", (uint8_t)imageBuffer[j * m_width + i]);
                }
                printf("\n");
            }

            printf("\n [%u] ===> cat(%u)\n\n", n, m_categoryBuffer[n]);
        }
    }

    int GetImageWidth() const
    {
        return m_width;
    }

    int GetImageHeight() const
    {
        return m_height;
    }

    int GetImageCount() const
    {
        return m_count;
    }

    int GetImageSize() const
    {
        return m_imageSize;
    }

    const float* GetImageData() const
    {
        return m_imageBuffer;
    }

    const uint8_t* GetCategoryData() const
    {
        return m_categoryBuffer;
    }

    //
    // Parse MNIST dataset
    // Specification of the dataset can be found in:
    // http://yann.lecun.com/exdb/mnist/
    //
    int Parse(const char* imageFile, const char* labelFile, bool verbose)
    {
        FILE* fimg = fopen(imageFile, "rb");
        if (!fimg)
        {
            printf("Failed to open %s for reading\n", imageFile);
            return 1;
        }

        FILE* flabel = fopen(labelFile, "rb");
        if (!flabel)
        {
            printf("Failed to open %s for reading\n", labelFile);
            return 1;
        }
        std::shared_ptr<void> autofimg(nullptr, [fimg, flabel](void*) {
            if (fimg) fclose(fimg);
            if (flabel) fclose(flabel);
        });

        uint32_t value;

        // Read magic number
        assert(!feof(fimg));
        if(fread(&value, sizeof(uint32_t), 1, fimg) != 1)
        {
            printf("Failed to read magic number from %s", imageFile);
            return 1;
        }

        assert(__builtin_bswap32(value) == 0x00000803);
        if (verbose)
            printf("Image Magic        :%0X%I32u\n",
                __builtin_bswap32(value), __builtin_bswap32(value));

        // Read count
        assert(!feof(fimg));
        if(fread(&value, sizeof(uint32_t), 1, fimg) != 1)
        {
            printf("Failed to read image count from %s", imageFile);
            return 1;
        }
        const uint32_t count = __builtin_bswap32(value);
        assert(count > 0);
        if (verbose)
            printf("Image Count        :%0X%I32u\n", count, count);

        // Read rows
        assert(!feof(fimg));
        if(fread(&value, sizeof(uint32_t), 1, fimg) != 1)
        {
            printf("Failed to read number of rows from %s", imageFile);
            return 1;
        }
        const uint32_t rows = __builtin_bswap32(value);
        assert(rows > 0);
        if (verbose)
            printf("Image Rows         :%0X%I32u\n", rows, rows);

        // Read cols
        assert(!feof(fimg));
        if(fread(&value, sizeof(uint32_t), 1, fimg) != 1)
        {
            printf("Failed to read number of columns from %s", imageFile);
        }
        const uint32_t cols = __builtin_bswap32(value);
        assert(cols > 0);
        if (verbose)
            printf("Image Columns      :%0X%I32u\n", cols, cols);

        // Read magic number (label)
        assert(!feof(flabel));
        if(fread(&value, sizeof(uint32_t), 1, flabel) != 1)
        {
            printf("Failed to read magic number from %s", labelFile);
            return 1;
        }
        assert(__builtin_bswap32(value) == 0x00000801);
        if (verbose)
            printf("Label Magic        :%0X%I32u\n",
                __builtin_bswap32(value), __builtin_bswap32(value));

        // Read label count
        assert(!feof(flabel));
        if(fread(&value, sizeof(uint32_t), 1, flabel) != 1)
        {
            printf("Failed to read label count from %s", labelFile);
            return 1;
        }
        // The count of the labels needs to match the count of the image data
        assert(__builtin_bswap32(value) == count);
        if (verbose)
            printf("Label Count        :%0X%I32u\n",
                __builtin_bswap32(value), __builtin_bswap32(value));

        Initialize(cols, rows, count);

        int counter = 0;
        while (!feof(fimg) && !feof(flabel) && counter < m_count)
        {
            float* imageBuffer = &m_imageBuffer[counter * m_imageSize];

            for (int j = 0; j < m_height; ++j)
            {
                for (int i = 0; i < m_width; ++i)
                {
                    uint8_t pixel;
                    if(fread(&pixel, sizeof(uint8_t), 1, fimg) != 1)
                    {
                        printf("Failed to read pixel (%d, %d) from image %d in %s",
                            i, j, counter, imageFile);
                        return 1;
                    }
                    imageBuffer[j * m_width + i] = pixel;
                }
            }

            uint8_t cat;
            fread(&cat, sizeof(uint8_t), 1, flabel);
            // assert(cat >= 0 && cat < c_categoryCount);
            m_categoryBuffer[counter] = cat;

            ++counter;
        }

        return 0;
    }
private:
    void Initialize(const int width, const int height, const int count)
    {
        m_width = width;
        m_height = height;
        m_imageSize = m_width * m_height;
        m_count = count;

        m_buffer = (float*)malloc(m_count * m_imageSize * sizeof(float));
        m_imageBuffer = m_buffer;
        m_categoryBuffer = (uint8_t*)malloc(m_count * sizeof(uint8_t));
    }

    // The total number of images
    int m_count;

    // Dimension of the image data
    int m_width;
    int m_height;
    int m_imageSize;

    // The entire buffers that stores both the image data and the category data
    float* m_buffer;

    // Buffer of images
    float* m_imageBuffer;

    // 1-of-N label of the image data (N = 10)
    uint8_t* m_categoryBuffer;

    static const int c_categoryCount = 10;
};

/**
 * Reads a dataset of images stored in an MNIST-style format from the specified
 * image and label files, and stores it in buffers that live in memory.
 *
 * @param image_fname name of file containing image data
 * @param label_fname name of file containing label data
 * @param n_images number of images in data set (set by reference)
 * @param c number of channels in each image in data set (set by reference)
 * @param h height of each image in data set (set by reference)
 * @param w width of each image in data set (set by reference)
 * @param n_classes number of clases in data set (set by reference)
 * @param image_data pointer to buffer that will contain loaded image data
 * @param label_data pointer to buffer that will contain loaded label data
 */
void LoadMNISTData(std::string image_fname, std::string label_fname,
    int &n_images, int &c, int &h, int &w, int &n_classes,
    float **image_data, float **label_data, bool verbose = true)
{
    // Parse the image and label files
    MNISTDataset data;
    assert(data.Parse(image_fname.c_str(), label_fname.c_str(), verbose) == 0);

    // Get metadata from image set and use it to set output variables
    n_images = data.GetImageCount();
    c = 1;
    h = data.GetImageHeight();
    w = data.GetImageWidth();
    n_classes = 10;

    // Normalize the images' pixel intensities to be between 0 and 1
    int X_size = n_images * c * h * w;
    *image_data = new float[X_size];
    for (int i = 0; i < X_size; ++i)
        (*image_data)[i] = data.GetImageData()[i] / 255.0f;

    // Convert the labels into one-hot vectors
    int Y_size = n_images * n_classes;
    *label_data = new float[Y_size];
    std::fill(*label_data, *label_data + Y_size, 0.0f);
    for (int i = 0; i < n_images; ++i)
    {
        int y = data.GetCategoryData()[i];
        (*label_data)[i * n_classes + y] = 1.0f;
    }
}
