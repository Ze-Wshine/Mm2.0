//Just a example for testing
#include <stdio.h>
#include <math.h>
#define ONE 1

#if 0
//nothing
#endif

#if 1
//highlight
#endif

union Node{
    int a;
    char b;
};

int main(){
   union Node node;
    node.a = 1;
    if(node.b == 0)
        printf("Big Endian");
    else
        printf("Little Endian");
    return 0;
}

