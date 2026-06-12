#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

VideoCapture carregarVideo(const char* nomeArquivo);
void processarVideo(VideoCapture& capture, int filtro);

int main(int argc, char** argv)
{
    int opcao;

    if (argc < 2)
    {
        printf("Uso: %s <video>\n", argv[0]);
        return 1;
    }

    while (true)
    {
        printf("\n===== FILTROS =====\n");
        printf("1 - Escala de Cinza\n");
        printf("2 - Emboss\n");
        printf("3 - Sharpen\n");
        printf("4 - Sobel\n");
        printf("0 - Sair\n");
        printf("Opcao: ");

        scanf("%d", &opcao);

        if (opcao == 0)
            break;

        VideoCapture capture = carregarVideo(argv[1]);

        if (!capture.isOpened())
        {
            printf("Erro ao abrir video\n");
            continue;
        }

        processarVideo(capture, opcao);

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

void aplicarSharpen(Mat& frame)
{
    Mat gray;
    cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    Mat resultado = gray.clone();

    for (int y = 1; y < gray.rows - 1; y++)
    {
        for (int x = 1; x < gray.cols - 1; x++)
        {
            int centro   = gray.at<uchar>(y, x);
            int cima     = gray.at<uchar>(y - 1, x);
            int baixo    = gray.at<uchar>(y + 1, x);
            int esquerda = gray.at<uchar>(y, x - 1);
            int direita  = gray.at<uchar>(y, x + 1);

            int valor =
                centro * 5
                - cima
                - baixo
                - esquerda
                - direita;

            if (valor < 0)
                valor = 0;

            if (valor > 255)
                valor = 255;

            resultado.at<uchar>(y, x) = valor;
        }
    }

    cvtColor(resultado, frame, cv::COLOR_GRAY2BGR);
}

void processarVideo(VideoCapture& capture, int filtro)
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
                aplicarSharpen(frame);
                break;

            case 3:
                printf("Sharpen ainda nao implementado\n");
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

