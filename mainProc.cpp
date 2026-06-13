#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;

VideoCapture carregarVideo(const char* nomeArquivo);
void processarVideo(VideoCapture& capture, int filtro, int intensidade);
void aplicarConvolucao(Mat& frame, const float kernel[3][3]);
void aplicarSharpen(Mat& frame, int intensidade);

int main(int argc, char** argv)
{
    int opcao;
    int intensidadeDoEfeito = 5;

    if (argc < 2)
    {
        printf("Uso: %s <video>\n", argv[0]);
        return 1;
    }

    while (true)
    {
        printf("\n===== FILTROS =====\n");
        printf("1 - Escala de Cinza\n");
        printf("2 - Sharpen\n");
        printf("3 - Emboss\n");
        printf("4 - Sobel\n");
        printf("0 - Sair\n");
        printf("Opcao: ");

        scanf("%d", &opcao);

        if (opcao == 0)
            break;

        if (opcao == 2)
        {
            printf("Qual intensidade? ");
            scanf("%d", &intensidadeDoEfeito);
        }

        VideoCapture capture = carregarVideo(argv[1]);

        if (!capture.isOpened())
        {
            printf("Erro ao abrir video\n");
            continue;
        }

        processarVideo(capture, opcao, intensidadeDoEfeito);

        capture.release();
    }

    return 0;
}

VideoCapture carregarVideo(const char* nomeArquivo)
{
    VideoCapture capture;
    capture.open(nomeArquivo);

    return capture;
}

void aplicarConvolucao(Mat& frame, const float kernel[3][3])
{
    Mat resultado = frame.clone();

    #pragma omp parallel for
    for (int y = 1; y < frame.rows - 1; y++)
    {
        const uchar* linhaCima  = frame.ptr<uchar>(y - 1);
        const uchar* linhaMeio  = frame.ptr<uchar>(y);
        const uchar* linhaBaixo = frame.ptr<uchar>(y + 1);

        uchar* dst = resultado.ptr<uchar>(y);

        for (int x = 1; x < frame.cols - 1; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                int idx = x * 3 + c;

                float valor = 0;

                valor += linhaCima[idx - 3]  * kernel[0][0];
                valor += linhaCima[idx]      * kernel[0][1];
                valor += linhaCima[idx + 3]  * kernel[0][2];

                valor += linhaMeio[idx - 3]  * kernel[1][0];
                valor += linhaMeio[idx]      * kernel[1][1];
                valor += linhaMeio[idx + 3]  * kernel[1][2];

                valor += linhaBaixo[idx - 3] * kernel[2][0];
                valor += linhaBaixo[idx]     * kernel[2][1];
                valor += linhaBaixo[idx + 3] * kernel[2][2];

                dst[idx] = (uchar)std::clamp((int)valor, 0, 255);
            }
        }
    }

    resultado.copyTo(frame);
}

void aplicarSharpen(Mat& frame, int intensidade)
{
    float kernel[3][3] =
    {
        {  0, -1,  0 },
        { -1, (float)intensidade, -1 },
        {  0, -1,  0 }
    };

    aplicarConvolucao(frame, kernel);
}

void processarVideo(VideoCapture& capture, int filtro, int intensidadeDoEfeito)
{
    Mat frame;

    while (capture.read(frame))
    {
        switch (filtro)
        {
            case 1:
                cvtColor(frame, frame, COLOR_BGR2GRAY);
                cvtColor(frame, frame, COLOR_GRAY2BGR);
                break;

            case 2:
                aplicarSharpen(frame, intensidadeDoEfeito);
                break;

            case 3:
                printf("Emboss ainda nao implementado\n");
                break;

            case 4:
                printf("Sobel ainda nao implementado\n");
                break;
        }

        imshow("Video", frame);

        if (waitKey(30) == 27)
            break;
    }

    destroyAllWindows();
}