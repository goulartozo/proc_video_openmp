#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <mpi.h>

using namespace cv;

VideoCapture carregarVideo(const char* nomeArquivo);
void processarVideo(VideoCapture& capture, int filtro, int intensidade);
void aplicarConvolucao(Mat& frame, const float kernel[3][3]);
void aplicarSharpen(Mat& frame, int intensidade);
void aplicarEscalaDeCinza(Mat& frame);
void aplicarEmboss(Mat& frame, int intensidade);
void aplicarSobel(Mat& frame, int intensidade);

void master(const char* nomeArquivo, int filtro, int intensidade, int size);
void worker(int rank, int filtro, int intensidade);

int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("\nProcesso %d de %d inicializado.\n", rank, size);

    if (argc < 2)
    {
        if (rank == 0)
        {
            printf("\nUso: %s <video> Ex: ./mainProc video2.avi\n", argv[0]);
        }

        MPI_Finalize();
        return 0;
    }

    if (size < 2)
    {
        if (rank == 0)
        {
            printf("\n Execute com pelo menos 2 processos.\n");
            printf("Ex: mpirun -np 4 ./mainProc video2.avi\n");
        }

        MPI_Finalize();
        return 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int opcao = 1;
    int intensidadeDoEfeito = 5;

    if(rank == 0)
    {
        printf("\n===== FILTROS =====\n");
        printf("1 - Escala de Cinza\n");
        printf("2 - Sharpen\n");
        printf("3 - Emboss\n");
        printf("4 - Sobel\n");
        printf("0 - Sair\n");
        printf("Opcao: ");
        fflush(stdout);

        scanf("%d", &opcao);

        if (opcao == 0){
            printf("\nEncerrando o programa...\n");
        }
        else if(opcao < 1 || opcao > 4)
		{
			printf("\nOpção inválida! Tente novamente.\n");
			opcao = 0;
		}
        else if (opcao != 1)
        {
            printf("\nQual intensidade? ");
            fflush(stdout);
            scanf("%d", &intensidadeDoEfeito);
        }
    }

    MPI_Bcast(&opcao, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&intensidadeDoEfeito, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (opcao == 0)
    {
        MPI_Finalize();
        return 0;
    }

    printf("\nProcesso %d recebeu configuração -> Filtro: %d | Intensidade: %d\n", rank, opcao, intensidadeDoEfeito);

    if (rank == 0)
    {
        printf("\nProcesso Master iniciando processamento do vídeo...\n");
        master(argv[1], opcao, intensidadeDoEfeito, size);
    }
    else
    {
        printf("Processo Worker %d aguardando blocos de imagem...\n", rank);
        worker(rank, opcao, intensidadeDoEfeito);
    }

    printf("Processo %d finalizado.\n", rank);
    MPI_Finalize();

    return 0;
}

void master(const char* nomeArquivo, int filtro, int intensidade, int size)
{
    VideoCapture capture(nomeArquivo);

    if (!capture.isOpened())
    {
        printf("\nErro ao abrir o vídeo.\n");
        int fimErro[3] = {0, 0, 0};
        MPI_Bcast(fimErro, 3, MPI_INT, 0, MPI_COMM_WORLD);
        return;
    }

    printf("\nVídeo aberto com sucesso.\n");

    Mat frame;

    while (capture.read(frame))
    {
        int largura = frame.cols;
        int altura = frame.rows;
        int canais = frame.channels();

        int info[3];
        info[0] = largura;
        info[1] = altura;
        info[2] = canais;

        MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);

        int workers = size - 1;

        int linhasBase = altura / workers;
        int resto = altura % workers;

        int linhaInicial = 0;

        // Envia blocos para os workers
        for (int destino = 1; destino < size; destino++)
        {
            int linhas = linhasBase;

            if (destino <= resto)
                linhas++;

            int primeiraLinha = linhaInicial;
            int ultimaLinha = linhaInicial + linhas - 1;

            int inicioEnvio = primeiraLinha;
            int fimEnvio = ultimaLinha;

            if (inicioEnvio > 0)
                inicioEnvio--;

            if (fimEnvio < altura - 1)
                fimEnvio++;

            int linhasEnviar = fimEnvio - inicioEnvio + 1;

            int dados[3];
            dados[0] = primeiraLinha;
            dados[1] = linhas;
            dados[2] = linhasEnviar;

            MPI_Send(dados, 3, MPI_INT, destino, 0, MPI_COMM_WORLD);

            MPI_Send(frame.ptr(inicioEnvio), linhasEnviar * largura * canais, MPI_UNSIGNED_CHAR, destino, 1, MPI_COMM_WORLD);

            printf("Master enviou %d linhas para Worker %d.\n", linhasEnviar, destino);

            linhaInicial += linhas;
        }

        // Recebe os blocos processados
        linhaInicial = 0;

        for (int origem = 1; origem < size; origem++)
        {
            int linhas = linhasBase;

            if (origem <= resto)
                linhas++;

            MPI_Recv(frame.ptr(linhaInicial), linhas * largura * canais, MPI_UNSIGNED_CHAR, origem, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("Master recebeu %d linhas do Worker %d.\n", linhas, origem);

            linhaInicial += linhas;
        }

        imshow("Video MPI + OpenMP", frame);

        if ((waitKey(30) & 0xFF) == 27)
            break;
    }

    int fim[3] = {0, 0, 0};

    MPI_Bcast(fim, 3, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Master finalizando processamento.\n");

    capture.release();

    destroyAllWindows();
}

void worker(int rank, int filtro, int intensidade)
{
    while (true)
    {
        int info[3];
        MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);

        int largura = info[0];
        int altura  = info[1];
        int canais  = info[2];

        // Sinal de término enviado pelo master ({0,0,0})
        if (largura == 0 && altura == 0 && canais == 0)
        {
            printf("Worker %d recebeu sinal de término.\n", rank);
            break;
        }

        int dados[3];
        MPI_Recv(dados, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int primeiraLinha = dados[0];
        int linhas        = dados[1];
        int linhasEnviar  = dados[2];

        // Bloco recebido já inclui as linhas de halo (se existirem)
        Mat bloco(linhasEnviar, largura, CV_8UC3);

        MPI_Recv(bloco.ptr(0), linhasEnviar * largura * canais, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        switch (filtro)
        {
            case 1: aplicarEscalaDeCinza(bloco); break;
            case 2: aplicarSharpen(bloco, intensidade); break;
            case 3: aplicarEmboss(bloco, intensidade); break;
            case 4: aplicarSobel(bloco, intensidade); break;
            default:
                printf("\nOpção inválida! Tente novamente.\n");
                break;
        }

        // Se primeiraLinha > 0, o bloco tem 1 linha de halo no topo
        int offsetTopo = (primeiraLinha == 0) ? 0 : 1;

        // Devolve só as linhas "reais" (sem o halo)
        MPI_Send(bloco.ptr(offsetTopo), linhas * largura * canais, MPI_UNSIGNED_CHAR,
                   0, 2, MPI_COMM_WORLD);

        printf("Worker %d processou e enviou %d linhas de volta ao Master.\n", rank, linhas);
    }
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
        { -1, 4.0f + (float)intensidade, -1 },
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