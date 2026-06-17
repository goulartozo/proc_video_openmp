#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;

VideoCapture carregarVideo(const char* nomeArquivo);
void processarVideo(VideoCapture& capture, int filtro, int intensidade);
void aplicarConvolucao(Mat& frame, const float kernel[3][3]);
void aplicarSharpen(Mat& frame, int intensidade);
void aplicarEscalaDeCinza(Mat& frame);

int main(int argc, char** argv)
{
    int opcao;
    int intensidadeDoEfeito = 5;

    if (argc < 2)
    {
        printf("Uso: %s <video> Ex: ./mainProc video2.avi\n", argv[0]);
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

        if(opcao > 4)
		{
			printf("\nOpção inválida! Tente novamente.\n");
			continue;
		}

        if (opcao != 1)
        {
            printf("\nQual intensidade? ");
            scanf("%d", &intensidadeDoEfeito);
        }

        VideoCapture capture = carregarVideo(argv[1]);

        if (!capture.isOpened())
        {
            printf("\n\nErro ao abrir video: %s\n", argv[1]);
            printf("Verifique se o arquivo existe e se o caminho está correto.\n");
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

void aplicarEmboss(Mat& frame, int i)
{
	float kernel[3][3] =
	{
		{-2 * (float)i, -1 * (float)i, 0},
		{-1 * (float)i, 1, 1 * (float)i},
		{0, 1 * (float)i, 2 * (float)i}	
	};
	
	aplicarConvolucao(frame, kernel);
}

void aplicarSobel(Mat& frame, int i)
{
	float primeiro_kernel[3][3] =
	{
		{-1 * (float)i, 0, 1 * (float)i},
		{-2 * (float)i, 0, 2 * (float)i},
		{-1 * (float)i, 0, 1 * (float)i}
	};
	
	aplicarConvolucao(frame, primeiro_kernel);
	
	float segundo_kernel[3][3] =
	{
		{1 * (float)i, 2 * (float)i, 1 * (float)i},
		{0, 0, 0},
		{-1 * (float)i, -2 * (float)i, -1 * (float)i}
	};
	
	aplicarConvolucao(frame, segundo_kernel);
}

void aplicarEscalaDeCinza(Mat& frame)
{
    #pragma omp parallel for // utilizado apenas for (sem guidance) porque a carga é uniforme entre iterações
    for (int y = 0; y < frame.rows; y++)
    {
        const uchar* src = frame.ptr<uchar>(y);
        uchar* dst = frame.ptr<uchar>(y);

        for (int x = 0; x < frame.cols; x++)
        {
            int idx = x * 3; 

            uchar b = src[idx]; // canal azul
            uchar g = src[idx + 1]; // canal verde
            uchar r = src[idx + 2]; // canal vermelho

            uchar cinza = (uchar)((114 * b + 587 * g + 299 * r) / 1000); // fórmula de conversão para escala de cinza

            dst[idx]     = cinza; 
            dst[idx + 1] = cinza;
            dst[idx + 2] = cinza;
        }
    }
}

void processarVideo(VideoCapture& capture, int filtro, int intensidadeDoEfeito)
{
    Mat frame;

    while (capture.read(frame))
    {
        switch (filtro)
        {
            case 1:
                aplicarEscalaDeCinza(frame);
                break;

            case 2:
                aplicarSharpen(frame, intensidadeDoEfeito);
                break;

            case 3:
                aplicarEmboss(frame, intensidadeDoEfeito);
                break;

            case 4:
                aplicarSobel(frame, intensidadeDoEfeito);
                break;

            default:
				printf("\nOpção inválida! Tente novamente.\n");
				break;
        }

        imshow("Video", frame);

        if (waitKey(30) == 27)
            break;
    }

    destroyAllWindows();
}