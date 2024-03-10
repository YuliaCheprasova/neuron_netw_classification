#include <iostream>
#include <ctime>
#include <cmath>
#include <fstream>
#include <string>

using namespace std;


//сигмоидальная функция, на вход s, на выходе y
void sigma(double** s, double** y, int numneuron_current, int L_current)
{
    y[L_current][numneuron_current] = 1./(1.+exp(-s[L_current][numneuron_current]));
}

//cчитаются s как сумма входов, умноженных на веса
void multmatrix(double*** w, double** u, double** s, int* numneuron, int current_m, int L, double** b, double** y)
{
    int i, j, k;
    double s_temp;
    // для первого слоя
    for (i = 0; i < numneuron[1]; i++)
    {
        s_temp = 0;
        for (j = 0; j < numneuron[0]; j++)//собирается один нейрон s[k][i]
        {
            s_temp += w[0][i][j]*u[current_m][j];
        }
        s[0][i]=s_temp + b[0][i];
        sigma(s, y, i, 0);
    }
    // для промежуточных слоев
    for (k = 1; k < L+1; k++)
    {
        for (i = 0; i < numneuron[k+1]; i++)
        {
            s_temp = 0;
            for (j = 0; j < numneuron[k]; j++)//собирается один нейрон s[k][i]
            {
                s_temp += w[k][i][j]*y[k-1][j];
            }
            s[k][i]=s_temp + b[k][i];
            if (k!=L){sigma(s, y, i, k);}
        }
    }
}



int main()
{
    ofstream fout("Qmeans.txt");
    ifstream iris("iris34_train.txt");
    setlocale(LC_ALL, "Rus");
    srand(time(0));
    int M=105, n, i, j, k, m, numiter = 1000, numenter = 1, numoutput = 3, L = 2; //(2 скрытых, 1 входной, 1 выходной, но веса при этом считаются три раза)
    //M-объем выборки, numenter-кол.входов, numneuron - кол.нейронов, n - номер итерации, m - номер наблюдения,
    double h = 0.1, e_temp, Q_temp, step, st = -0.3, fin = 0.5, summ_exp;
    string line;



    //0 индекс - количество u, 1 - кол-во нейронов в 1 скрытом слое, последний - количество нейронов на выходе
    //cout << "Количество скрытых слоев ";
    //cin >> L;
    int *numneuron = new int[L+2];//1 5 3 1
    /*cout << "Количество входов ";
    cin >> numneuron[0];
    for (i = 1; i < L+1; i++)
    {
        cout << "Количество нейронов в " << i << " скрытом слое ";
        cin >> numneuron[i];
    }
    cout << "Количество выходов ";
    cin >> numneuron[L+1];*/
    numneuron[0]=2;
    numneuron[1]=8;
    numneuron[2]=4;
    numneuron[L+1]=numoutput;// в выходном слое один "нейрон"

    double** data = new double* [105];
    for (i = 0; i < 105; i++)
    {
        data[i] = new double[3];
    }
    double* y_last = new double [numoutput];
    double **d = new double* [M];
    for(i =0; i < M; i++)
    {
        d[i] = new double [numneuron[L+1]];
    }
    //y столько же сколько нейронов на каждом слое (в выходном слое считается только s)
    double **y = new double* [L];
    for(i = 0; i < L; i++)
    {
        y[i] = new double [numneuron[i+1]];
    }
    //в каждом слое матрица весов по строчкам выходы слоя, по столбцам входы, веса считаются на один раз больше чем слоев в сети
    double*** w = new double**[L+1];
    for (i = 0; i < L+1; i++)
    {
        w[i] = new double*[numneuron[i+1]];
        for (j = 0; j < numneuron[i+1]; j++)
        {
            w[i][j] = new double[numneuron[i]];
        }
    }
    // их столько же сколько весов
    double*** wgrad = new double**[L+1];
    for (i = 0; i < L+1; i++)
    {
        wgrad[i] = new double*[numneuron[i+1]];
        for (j = 0; j < numneuron[i+1]; j++)
        {
            wgrad[i][j] = new double[numneuron[i]];
        }
    }
    // как я поняла u используется только на первом слое
    double **u = new double* [M];
    for (i = 0; i < M; i++)
    {
        u[i] = new double[numneuron[0]];
    }
    //s столько же сколько нейронов на каждом слое + в выходном слое 1 раз посчитать s
    double **s = new double*[L+1];
    for(i = 0; i < L+1; i++)
    {
        s[i] = new double [numneuron[i+1]];
    }
    double **e = new double*[L+1];
    for(i = 0; i < L+1; i++)
    {
        e[i] = new double [numneuron[i+1]];
    }
    double **delta = new double*[L+1];
    for(i = 0; i < L+1; i++)
    {
        delta[i] = new double [numneuron[i+1]];
    }
    double **b = new double*[L+1];
    for(i = 0; i < L+1; i++)
    {
        b[i] = new double [numneuron[i+1]];
    }
    double **bgrad = new double*[L+1];
    for(i = 0; i < L+1; i++)
    {
        bgrad[i] = new double [numneuron[i+1]];
    }


    // начальные значения переменных
    for (i = 0; i < 105; i++)
    {
        for (j = 0; j < 3; j++)
        {
            iris >> data[i][j];
        }
    }
    for(k = 0; k < L+1; k++)
    {
        for(i = 0; i < numneuron[k+1]; i++)
        {
            for(j = 0; j < numneuron[k]; j++)
            {
                w[k][i][j] = rand() % 3 - 1;//от -1 до 1
            }
        }
    }
    for (k = 0; k < L+1; k++)
    {
        for (i = 0; i < numneuron[k+1]; i++)
        {
            b[k][i] = 0;
        }
    }
    for(i = 0; i < M; i++)
    {
        for(j = 0; j < numneuron[0]; j++)
        {
            u[i][j] = data[i][j];
            //fout << u[i][j]<< "\n";
        }
    }
    //fout << "\n";
    // создание синтетических данных
    for(i = 0; i < M; i++)
    {
        for(j = 0; j < numneuron[L+1]; j++)
        {

            d[i][j] = 0;
            //fout << d[i][j] << "\n";
        }
    }
    for(i = 0; i < M; i++)
    {
        for(j = 0; j < numoutput; j++)
        {
            if(data[i][2]==j)
            {
                d[i][j]=1;
            }
            cout << d[i][j] << " ";//проверить тут надо
        }
        cout << endl;
    }
    //fout << "\n";


    // основной цикл

    for(n=0; n<numiter; n++)
    {
        cout << "Iteration " << n << endl;
        Q_temp = 0;

        for (m = 0;m < M; m++)// сейчас он у меня как будто пытается подстроиться под все наблюдения, и только потом идет на следующую итерацию
        {
            summ_exp = 0;
            // расчет s и y
            multmatrix(w, u, s, numneuron, m, L, b, y);
            for(i = 0; i < numneuron[L+1]; i++)
            {
                summ_exp+=exp(s[L][i]);// найти максимальное s и вычесть это значение из всех s, посчитать accuracy
            }
            // для выходного слоя
            for(i = 0; i < numneuron[L+1]; i++)
            {
                y_last[i] = exp(s[L][i])/summ_exp;
                e[L][i] = y_last[i]-d[m][i];
                delta[L][i] = e[L][i];
            }
            for(k = L-1; k>=0; k--)
            {
                for(i = 0; i < numneuron[k+1]; i++)//3 в сущности это вход этого слоя
                {
                    e_temp = 0;
                    for(j = 0; j < numneuron[k+2]; j++)//в сущности это выход
                    {
                        e_temp += w[k+1][j][i]*delta[k+1][j];
                    }
                    e[k][i] = e_temp;
                    delta[k][i]=e[k][i]*(1-y[k][i])*y[k][i];
                }
            }
            for(k = 0; k < L+1; k++)
            {
                for(i = 0; i < numneuron[k+1]; i++)//5
                {
                    bgrad[k][i] = delta[k][i];
                    b[k][i] = b[k][i]-h*bgrad[k][i];
                    for(j = 0; j < numneuron[k]; j++)//1
                    {
                        if (k==0)
                        {
                            wgrad[k][i][j] = u[m][j]*delta[k][i];
                        }
                        if (k!=0)
                        {
                            wgrad[k][i][j] = y[k-1][j]*delta[k][i];
                        }
                        w[k][i][j] = w[k][i][j]-h*wgrad[k][i][j];
                    }
                }
            }
            for(i = 0; i < numneuron[L+1]; i++)
            {
                 if(n!=numiter-1)
                {
                    Q_temp-=log(y_last[i])*d[m][i];
                }
                if(n==numiter-1)
                {
                    multmatrix(w, u, s, numneuron, m, L, b, y);
                    Q_temp-=log(y_last[i])*d[m][i];
                    //fout << s[2][0] << "\n";
                }
            }
        }
        cout << Q_temp/M << endl;
        //fout << Q_temp/M << "\n";
    }


    for (i = 0; i < M; i++)
    {
        delete [] d[i];
    }
    delete[] d;
    for (i = 0; i < L+1; i++)
    {
        for (j = 0; j < numneuron[i+1]; j++)
        {
            delete[] w[i][j];
        }
        delete[] w[i];
    }
    delete[] w;
    for (i = 0; i < L+1; i++)
    {
        for (j = 0; j < numneuron[i+1]; j++)
        {
            delete[] wgrad[i][j];
        }
        delete[] wgrad[i];
    }
    delete[] wgrad;
    for (i = 0; i < M; i++)
    {
        delete [] u[i];
    }
    delete[] u;
    for (i = 0; i < L+1; i++)
    {
        delete [] s[i];
    }
    delete[] s;
    for (i = 0; i < L; i++)
    {
        delete [] y[i];
    }
    delete[] y;
    for (i = 0; i < L+1; i++)
    {
        delete [] e[i];
    }
    delete[] e;
    for (i = 0; i < L+1; i++)
    {
        delete [] delta[i];
    }
    delete[] delta;
    for (i = 0; i < L+1; i++)
    {
        delete[] bgrad[i];
    }
    delete[] bgrad;
    for (i = 0; i < L+1; i++)
    {
        delete[] b[i];
    }
    delete[] b;
    for (i = 0; i < 105; i++)
    {
        delete[] data[i];
    }
    delete[] data;
    delete[] y_last;
    delete[] numneuron;

    fout.close();
    iris.close();
}
