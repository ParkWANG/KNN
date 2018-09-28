//
//  main.cpp
//  KNN_CPU_FAST
//
//  Created by 王方 on 18/9/21.
//  Copyright © 2018年 王方. All rights reserved.
//

//
//  main.cpp
//  KNN_CPU
//
//  Created by 王方 on 18/9/18.
//  Copyright © 2018年 王方. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "Countime.h"
#include <algorithm>

using namespace std;
#define DatasetVolume 269648
#define KNN_Num 10
#define FREQUENCY 3200000000
#define QUERYSUM 10

struct ObjectInData{
    int TagNO;
    double L2NormBound;
};

struct ResultArray{
    int TagNO;
    double L2NormExact;
};

struct  ObjectInData DataObject[DatasetVolume];

struct  ResultArray Resultarray[KNN_Num];

bool Cmpare(const ObjectInData &a, const ObjectInData &b)            //const必须加，不然会错，目前不懂为啥。当return的是ture时，a先输出，所以示例中是升序
{
    return a.L2NormBound<b.L2NormBound;
}

bool Cmpare_R(const ResultArray &a, const ResultArray &b)
{
    return a.L2NormExact<b.L2NormExact;
}

double Exact_L2Norm_POINT_ (int Object_no, double Query_element[144],int Integer_bits);

double Exact_L2Norm_POINT_FAST_ (int Object_no, double Query_element[144],int Integer_bits);

int main(int argc, const char * argv[]) {
    
    
    //*--------------------------------------------------SET PARAMETERS    --------------------------------------------*/
    int Integer_bits=3    ;
    
    double Global_time;
    struct timeval tv1,tv2;
    //*--------------------------------------------------QUERY POINT CATCH--------------------------------------------*/
    //*---------------------------------------------------------------------------------------------------------------*/
    ifstream infile_query("/Users/wangfang/Documents/KNN/Query_Data_100-1.txt");
    double Query_Original_element[100][144]={0};
    double Query_Modifiyed_element[100][144]={0};
    double Query_Integer_element[100][144]={0};
    double Query_SUM_POW2[100]={0};
    double Query_SUM_POW1[100]={0};
    int Bias=pow(10, Integer_bits);
    string temp_query;
    int No_query_line=0;
    while(getline(infile_query,temp_query))
    {
        vector<string> arr1;
        istringstream ss(temp_query);
        string word;
        while(ss>>word) {
            arr1.push_back(word);
        }
        
        for(size_t i=0; i<arr1.size(); i++) {
            
            Query_Original_element[No_query_line][i]= stod(arr1[i]);
            Query_Modifiyed_element[No_query_line][i]=Query_Original_element[No_query_line][i]*pow(10, Integer_bits)+Bias;
            Query_Integer_element[No_query_line][i]=(long int)Query_Modifiyed_element[No_query_line][i];
            Query_SUM_POW2[No_query_line]+=pow(Query_Modifiyed_element[No_query_line][i], 2);
            Query_SUM_POW1[No_query_line]+=Query_Integer_element[No_query_line][i];
        }
        No_query_line++;
    }
    
    printf("****************STEP TWO***************\n");
    
    //*---------------------------------------COMPUTE LOWER BOUND OF L2-NORM------------------------------------------*/
    //*---------------------------------------------------------------------------------------------------------------*/
    int Prune_count_average=0;
    int Update_count_average=0;
    
    for(int query_no=0; query_no<QUERYSUM; query_no++){
        
        ifstream infile_object("/Users/wangfang/Documents/KNN/Low_Level_Features/Normalized_CORR.dat");
        double Object_SUM_POW1=0;
        double Object_SUM_POW2=0;
        double Object_SUM_InnerProdcut=0;
        double Lower_Bound_L2Norm[DatasetVolume]={0};
        int Object_lineNo=0;
        string temp_object;
        while(getline(infile_object,temp_object))
        {
            Object_SUM_InnerProdcut=0;
            Object_SUM_POW1=0;
            Object_SUM_POW2=0;
            
            vector<string> arr1;
            istringstream ss(temp_object);
            string word;
            while(ss>>word) {
                arr1.push_back(word);
            }
            
            for(size_t i=0; i<arr1.size(); i++) {
                
                double Object_Orginal_element= stod(arr1[i]);
                double Object_Modified_element=Object_Orginal_element*pow(10, Integer_bits)+pow(10, Integer_bits);
                double Object_Integer_element=(long int)Object_Modified_element;
                Object_SUM_POW1+=Object_Integer_element;
                Object_SUM_POW2+=pow(Object_Modified_element, 2);
                Object_SUM_InnerProdcut += Object_Integer_element*Query_Integer_element[query_no][i];
            }
            
            Lower_Bound_L2Norm[Object_lineNo]=Object_SUM_POW2 +Query_SUM_POW2[query_no]-2*(Object_SUM_InnerProdcut +Object_SUM_POW1 +Query_SUM_POW1[query_no] +1);
            
            Object_lineNo++;
            
        }
        Global_time+=double(DatasetVolume*4*Integer_bits/8)/1024/1024/1000;
        
        //*---------------------------------------SORT K SAMLLEST LOWER BOUND------------------------------------------*/
        for(int k=0; k<DatasetVolume; k++)
        {
            DataObject[k].TagNO=k;
            DataObject[k].L2NormBound=Lower_Bound_L2Norm[k];
        }
        gettimeofday(&tv1,NULL);
        sort(DataObject, DataObject+DatasetVolume, Cmpare);
        gettimeofday(&tv2,NULL);
        double sort_time=tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec)/1000000.0;
        
        //*---------------------------------------DO EXACT COMPUTATION && PRUNING RANGE------------------------------------------*/
        
        
        for(int j=0; j<KNN_Num; j++)
        {
            int temp_tagno=DataObject[j].TagNO;
            Resultarray[j].TagNO=temp_tagno;
            Resultarray[j].L2NormExact=Exact_L2Norm_POINT_FAST_(temp_tagno, Query_Modifiyed_element[query_no], Integer_bits);
            printf("Orignal Array Result NO: %d %lf  \n",Resultarray[j].TagNO,Resultarray[j].L2NormExact);
        }
        
        /*****Test******/
        printf("%d Resr Array: %lf \n",KNN_Num-1,Resultarray[KNN_Num-1].L2NormExact);
        printf("%d Data Bound: %lf \n",KNN_Num-1,DataObject[KNN_Num].L2NormBound);
        
        for(int no=KNN_Num; no<DatasetVolume; no++)
        {
            if(Resultarray[KNN_Num-1].L2NormExact <= DataObject[no].L2NormBound )
                break;
            else
            {
                double L2NormExact_no=Exact_L2Norm_POINT_FAST_(DataObject[no].TagNO, Query_Modifiyed_element[query_no], Integer_bits);
                Prune_count_average++;
                if(L2NormExact_no < Resultarray[KNN_Num-1].L2NormExact)  //Update Array
                {
                    Resultarray[KNN_Num-1].TagNO=no;
                    Resultarray[KNN_Num-1].L2NormExact=L2NormExact_no;
                    sort(Resultarray, Resultarray+KNN_Num, Cmpare_R);
                    Update_count_average++;
                    printf("##################\n");
                }
            }
        }
        
        printf("Prune Range Average Count: %d\n",Prune_count_average);
        printf("Prune Range Average Count: %d\n",Update_count_average);
        
        for(int j=0; j<KNN_Num; j++)
        {
            printf("TagNO: %d  ",Resultarray[j].TagNO);
            printf("L2NormExact: %lf\n",Resultarray[j].L2NormExact);
        }
        
        Global_time+=sort_time;
        printf("Sort time: %lf\n", sort_time);
        printf("---------------NO Query: %d\n",query_no);
    }
    
    
    printf("Prune Range Average Count: %d\n",Prune_count_average/QUERYSUM);
    printf("Update Result Array Count: %d\n",Update_count_average/QUERYSUM);
    printf("Sort Array Count: %lf\n",Global_time/QUERYSUM);

    
    
}


double Exact_L2Norm_POINT_FAST_ (int Object_no, double Query_element[144],int Integer_bits){
    
    printf("The Exact computation object NO:%d \n",Object_no);
    double Object_L2Norm=0;
    FILE *fp=fopen("/Users/wangfang/Documents/KNN/Low_Level_Features/Normalized_CORR.dat", "r");
    char *p;
    if (NULL == fp) {
        printf("CAN NOT OPEN THE FILE\n");
        printf("CAN NOT OPEN THE FILE\n");
    }
    /*获取文件字节大小size*/
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if(size > 0)
    {
       // printf("size: %ld\n",size);
        p = (char *)calloc(size,sizeof(char));
    }
    /*读文件内容存入内存*/
    fread(p, size, 1, fp);
    fclose(fp);
    p[size-1] = '\0';
    
    int line_no=0;
    stringstream target_line;
    for(int k=0; k<size; k++)
    {
        if(p[k]=='\n')
        {   line_no++;
        }
        if(line_no==Object_no)
            target_line<<p[k];
    }

    vector<string> arr1;
    string word;
    while(target_line>>word) {
        arr1.push_back(word);
    }
    for(size_t i=0; i<arr1.size(); i++) {
        
        double Object_Orginal_element= stod(arr1[i]);
        double Object_Modified_element=Object_Orginal_element*pow(10, Integer_bits)+pow(10, Integer_bits);
        Object_L2Norm+= pow(Object_Modified_element-Query_element[i],2);
        
    }
    free(p);
    return Object_L2Norm;
}




double Exact_L2Norm_POINT_ (int Object_no, double Query_element[144],int Integer_bits){
    ifstream infile_object("/Users/wangfang/Documents/KNN/Low_Level_Features/Normalized_CORR.dat");
    printf("Object_NO need eact compu: %d \n",Object_no);
    double Object_L2Norm=0;
    string temp_object;
    int Line_No=0;
    while(getline(infile_object,temp_object))
    {
        
        vector<string> arr1;
        istringstream ss(temp_object);
        string word;
        while(ss>>word) {
            arr1.push_back(word);
        }
        
        if(Line_No==Object_no){
            for(size_t i=0; i<arr1.size(); i++) {
                double Object_Orginal_element= stod(arr1[i]);
                double Object_Modified_element=Object_Orginal_element*pow(10, Integer_bits)+pow(10, Integer_bits);
                Object_L2Norm+= pow(Object_Modified_element-Query_element[i],2);
            }
        }
        Line_No++;
        if(Line_No>Object_no)
            break;
    }
    
    return Object_L2Norm;
}



