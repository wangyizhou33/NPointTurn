#ifndef SAVE_PPM_H
#define SAVE_PPM_H

#include <stdio.h>

inline int savePPM(const unsigned char* img, int w, int h, const char* fname)
{
    int back;
    FILE* f;

    f = fopen(fname, "wb");

    if (f == NULL)
        return (1);

    back = fprintf(f, "P6 %d %d 255\n", w, h);
    if (back <= 0)
        return (2);

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            back = fputc(img[i * w + j], f);
            if (back == EOF)
                return (3);
            back = fputc(img[w * h + i * w + j], f);
            if (back == EOF)
                return (3);
            back = fputc(img[2 * w * h + i * w + j], f);
            if (back == EOF)
                return (3);
        }
    }
    back = fclose(f);
    if (back == EOF)
    {
        return (4);
    }

    return (0);
}

#endif //SAVE_PPM_H
