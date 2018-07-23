// Copyright 2018 ishan@khatri.io
// Based on MNISTParser.h by Henry Tan
// Original source: https://github.com/ht4n/CPPMNISTParser

#ifndef MNISTPARSER_H_
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <assert.h>

#include <vector>
#include <tuple>

#ifdef _WIN32
#include <Eigen/dense>
#endif

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#endif

using std::vector;
using Eigen::VectorXf;
using std::tuple;
using std::make_tuple;
using std::get;

typedef tuple<VectorXf, int> data;

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
        m_imageSize(0)
        // m_buffer(nullptr),
        // m_imageBuffer(nullptr),
        // m_categoryBuffer(nullptr)
    {
    }

    // ~MNISTDataset()
    // {
    //     if (m_buffer) free(m_buffer);
    //     if (m_categoryBuffer) free(m_categoryBuffer);
    // }

    void Print(int thing) {
        vector<data> thing_to_print;
        if(thing == 0){
            thing_to_print = training_data;
        }
        else if (thing == 1){
            thing_to_print = validation_data;
        }
        else{
            thing_to_print = test_data;
        }
        int n = 0;
        for (data d : thing_to_print) {
          VectorXf imageBuffer = get<0>(d);
            for (size_t j = 0; j < m_height; ++j)
            {
                for (size_t i = 0; i < m_width; ++i)
                {
                    printf("%3d ", (uint8_t)imageBuffer[j * m_width + i]);
                }
                printf("\n");
            }

            printf("\n [%d] ===> cat(%u)\n\n", n, get<1>(d));
            n++;
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

    vector<data> GetTrainingData() {
        return training_data;
    }

    vector<data> GetValidationData() {
        return validation_data;
    }

    vector<data> GetTestData() {
        return test_data;
    }

    //
    // Parse MNIST dataset
    // Specification of the dataset can be found in:
    // http://yann.lecun.com/exdb/mnist/
    //
    int Parse(const char* imageFile, const char* labelFile, bool test)
    {
        FILE* fimg = nullptr;
        if (0 != fopen_s(&fimg, imageFile, "rb"))
        {
            printf("Failed to open %s for reading\n", imageFile);
            return 1;
        }
        
        FILE* flabel = nullptr;
        if (0 != fopen_s(&flabel, labelFile, "rb"))
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
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, fimg);
        printf("Image Magic        :%0X(%I32u)\n", _byteswap_ulong(value), _byteswap_ulong(value));
        assert(_byteswap_ulong(value) == 0x00000803);

        // Read count
        assert(!feof(fimg));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, fimg);
        const uint32_t count = _byteswap_ulong(value);
        printf("Image Count        :%0X(%I32u)\n", count, count);
        assert(count > 0);

        // Read rows
        assert(!feof(fimg));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, fimg);
        const uint32_t rows = _byteswap_ulong(value);
        printf("Image Rows         :%0X(%I32u)\n", rows, rows);
        assert(rows > 0);

        // Read cols
        assert(!feof(fimg));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, fimg);
        const uint32_t cols = _byteswap_ulong(value);
        printf("Image Columns      :%0X(%I32u)\n", cols, cols);
        assert(cols > 0);

        // Read magic number (label)
        assert(!feof(flabel));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, flabel);
        printf("Label Magic        :%0X(%I32u)\n", _byteswap_ulong(value), _byteswap_ulong(value));
        assert(_byteswap_ulong(value) == 0x00000801);

        // Read label count
        assert(!feof(flabel));
        fread_s(&value, sizeof(uint32_t), sizeof(uint32_t), 1, flabel);
        printf("Label Count        :%0X(%I32u)\n", _byteswap_ulong(value), _byteswap_ulong(value));
        // The count of the labels needs to match the count of the image data
        assert(_byteswap_ulong(value) == count);

        Initialize(cols, rows, count);

        size_t counter = 0;
        while (!feof(fimg) && !feof(flabel) && counter < m_count)
        {
            VectorXf image_data = VectorXf(m_imageSize);

            for (size_t j = 0; j < m_height; ++j)
            {
                for (size_t i = 0; i < m_width; ++i)
                {
                    uint8_t pixel;
                    fread_s(&pixel, sizeof(uint8_t), sizeof(uint8_t), 1, fimg);

                    image_data[j * m_width + i] = pixel;
                }
            }

            uint8_t cat;
            fread_s(&cat, sizeof(uint8_t), sizeof(uint8_t), 1, flabel);
            // assert(cat >= 0 && cat < c_categoryCount);
            if(test){
                test_data.push_back(make_tuple(image_data, cat));
            }
            else if(counter < 50000){
                training_data.push_back(make_tuple(image_data, cat));
            }
            else{
                validation_data.push_back(make_tuple(image_data, cat));
            }
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
    }

    // The total number of images
    int m_count;

    // Dimension of the image data
    int m_width;
    int m_height;
    int m_imageSize;

    static const int c_categoryCount = 10;

    vector<data> training_data;
    vector<data> validation_data;
    vector<data> test_data;
};

#endif
