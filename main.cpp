/***********************************************
 Fisherian model of sexual selection on male courtship effort, based on 'conditional indicator' mechanism.
 Female preference for fix and flexible courtship behaviour on the fecundity model.
 July 2016, Exeter

 Added in the offspring that when the display is less than 0, reset alpha and beta to 0
***********************************************/

//HEADER FILES

#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>


// constants, type definitions, etc.

using namespace std;

const int N            = 5000;    // population size

const double mutPa     = 0.05;    // mutation rate in p for the fix display
const double stepPa    = 0.4;     // mutation step size in p for the fix display
const double mutPb     = 0.05;    // mutation rate in p for the flexible display
const double stepPb    = 0.4;     // mutation step size in p for the flexible display
const double mutPc     = 0.05;    // mutation rate in p for the interaction
const double stepPc    = 0.4;     // mutation step size in p for the interaction
const double mutAlpha  = 0.05;    // mutation rate in alpha
const double stepAlpha = 0.4;     // mutation step size in alpha
const double mutBeta   = 0.05;    // mutation rate in beta
const double stepBeta  = 0.4;     // mutation step size in beta
const double mutGamma  = 0.05;    // mutation rate in gamma
const double stepGamma = 0.4;     // mutation step size in gamma
const double mutC      = 0.05;    // mutation rate in male preference
const double mutV      = 0.00;    // mutation rate in v
const double stepV     = 0.0;     // mutation step size in v
const double biasGamma = 0.99;    // prob that mutation in gamma is downwards (0.5 = unbiased)

const double a1        = 1;       // strength of preference for the trait
const double a2        = 1;       // strength of preference for flexible display
const double b         = 0.0025;  // cost of preference
const double k         = 0.1;     // cost of fixed trait
const double c         = 0.2;     // cost of display
const double Vopt      = 10.0;    // optimal value of V

const double avNumComp = 3.0;     // number of competing males is drawn from Y~Poisson(avNumComp)

const double initPa    = 1.0;     // starting value of the female preference (in gen 0)
const double initPb    = 1.0;     // starting value of the female preference (in gen 0)
const double initPc    = 1.0;     // starting value of the female preference (in gen 0)
const double initAlpha = 0.0;     // starting value of alpha (in gen 0)
const double initBeta  = 0.0;     // starting value of beta (in gen 0)
const double initGamma = 0.0;     // starting value of gamma (in gen 0)
const double initV     = Vopt;    // starting value of v (in gen 0)
const double meanFec   = 5.0;     // mean female fecundity
const double stdevFec  = 3.0;     // standard deviation of mean female fecundity

const int NumGen       = 10000;   // number of generations
const int mutGen       = 2000;    // number of generations after which p mutates
const int survoffsp    = 10;       // min number of males and females x2 that survive until adulthood
const int survdisplay  = 20;       // min number of males that survive after display
const int numrep       = 20;      // number of replicates
const int skip         = 50;      // interval between print-outs

struct Individual // define individual traits
{
  double Pa0,Pa1,Pb0,Pb1,Pc0,Pc1,A0,A1,B0,B1,G0,G1,V0,V1, // genetic values for female and male preference, alpha, beta, viability
    pA,pB,pC,alpha,beta,gamma,v,vb,f,fCum,t;     // phenotypic values
  bool alive;                     // 1 alive, 0 dead
  int partner,nKids,C;              // number of offspring. Male preference
};

typedef Individual Population[N];

Population F0,F1,M0,M1;           // female and male parent and offspring generations
int NF,NM,SampleSize,gen,rep,
  killedMales;                    // number of females, males and number of males sampled by female
double sumFec,meanPa,meanPb,meanPc,meanV,
  meanAlpha,meanBeta,meanGamma,C0,C1,C2,
  meanAlpha1,meanBeta1,meanGamma1,meanAlpha2,meanBeta2,meanGamma2,
  stdevPa,stdevPb,stdevPc,stdevV,
  stdevAlpha,stdevBeta,stdevGamma,
  corr_vAlpha,corr_vBeta,corr_vGamma;         // population means, standard deviations and correlations

ofstream fout("display.txt");     // output file

// random numbers
mt19937 mt(time(0)); // random number generator
uniform_real_distribution<double> Uniform(0, 1); // real number between 0 and 1 (uniform)
poisson_distribution<int> Poisson(avNumComp); // random draw from Poisson distribution with mean avNumComp


/* INITIALISE POPULATION */
void Init()
{
	int i;
//	SET SEED need to turn back on
//  SetSeed((unsigned)time(NULL));

	for (i=0;i<N/2;i++) // for half of individuals in population
	  {
      F0[i].Pa0=initPa;
      F0[i].Pa1=initPa;
      F0[i].Pb0=initPb;
      F0[i].Pb1=initPb;
      F0[i].Pc0=initPc;
      F0[i].Pc1=initPc;
      F0[i].A0=initAlpha;
	  F0[i].A1=initAlpha;
	  F0[i].B0=initBeta;
	  F0[i].B1=initBeta;
	  F0[i].G0=initGamma;
	  F0[i].G1=initGamma;
	  F0[i].V0=Vopt;
	  F0[i].V1=Vopt;
	  F0[i].C=9999;

      M0[i].Pa0=initPa;
      M0[i].Pa1=initPa;
      M0[i].Pb0=initPb;
      M0[i].Pb1=initPb;
      M0[i].Pc0=initPc;
      M0[i].Pc1=initPc;
      M0[i].A0=initAlpha;
      M0[i].A1=initAlpha;
      M0[i].B0=initBeta;
      M0[i].B1=initBeta;
	  M0[i].G0=initGamma;
	  M0[i].G1=initGamma;
	  M0[i].V0=Vopt;
	  M0[i].V1=Vopt;
	  M0[i].C=0;
      }

	NF=NM=N/2; // females and males equally abundant in starting population
}



/* CALCULATE PHENOTYPES AND RESCALE FITNESS */
void Phenotypes()
{
	int i, deadMales = 0, deadFemales = 0;

	sumFec = 0.0; // wipe clean

	// Calculate female trait values and absolute viabilities
	for (i=0;i<NF;i++) // for all females
	{
		F0[i].pA = 0.5*(F0[i].Pa0+F0[i].Pa1); // pref-fix is average of maternal and paternal genes
		F0[i].pB = 0.5*(F0[i].Pb0+F0[i].Pb1); // pref-flexible is average of maternal and paternal genes
		F0[i].pC = 0.5*(F0[i].Pc0+F0[i].Pc1); // pref interaction is average of maternal and paternal genes
		F0[i].vb = 0.5*(F0[i].V0+F0[i].V1); // genetic viability is average of maternal and paternal genes
		F0[i].alpha = 0.5*(F0[i].A0 + F0[i].A1); //average of maternal and paternal genes
        F0[i].beta = 0.5*(F0[i].B0 + F0[i].B1); //average of maternal and paternal genes
		F0[i].v = exp(-b*F0[i].pA*F0[i].pA - b*F0[i].pB*F0[i].pB - b*F0[i].pC*F0[i].pC);  // female viability depends on expressed pref

        normal_distribution<double> Normal(meanFec, stdevFec); // random draw from normal distribution
		F0[i].f = Normal(mt); // female fecundity randomly drawn from normal distribution
		if (F0[i].f < 0.0) F0[i].f = 0.0; // reset to zero if negative

		// VIABILITY SELECTION
		F0[i].alive=1; // female is alive
		if ( deadFemales < NF-survoffsp && Uniform(mt) > F0[i].v )
		{
          F0[i].alive=0; // female does not survive to adulthood due to low viability
          F0[i].f=0.0; // zero fecundity
          deadFemales++;
		}
		sumFec += F0[i].f; // fecundity added to cumulative total (for use later)

	    F0[i].partner=9999; // identity of chosen male; initially blank (9999)
	    F0[i].C=9999; // this gene is only carried by males, so ignore it for females
	}

//	cout << "gen: " << gen << "   dead females: " << deadFemales << endl;

	// Rescale and make cumulative so that F0[NF-1].fCum=1 for surviving females
	F0[0].fCum = F0[0].f/sumFec;
	for (i=1;i<NF;i++) F0[i].fCum = F0[i-1].fCum+F0[i].f/sumFec;

	// Calculate male trait values
	for (i=0;i<NM;i++)
	{
	    M0[i].alpha = 0.5*(M0[i].A0 + M0[i].A1); //average of maternal and paternal genes
        M0[i].beta = 0.5*(M0[i].B0 + M0[i].B1); //average of maternal and paternal genes
        M0[i].gamma = 0.5*(M0[i].G0 + M0[i].G1); //average of maternal and paternal genes
		M0[i].vb = 0.5*(M0[i].V0 + M0[i].V1); // genetic viability is average of maternal and paternal genes
		M0[i].t = M0[i].gamma; // male fixed trait is a conditional indicator, i.e. expression depends on viability
		M0[i].v = exp(-k*M0[i].t*M0[i].t); // male viability depends on expressed trait

		// VIABILITY SELECTION
		M0[i].alive=1; // male is alive
		if ( deadMales < NM-survoffsp && Uniform(mt) > M0[i].v ) { M0[i].alive=0; deadMales++; }// up to XX% of males do not survive until adulthood due to low viability
	}

 //   cout << "gen: " << gen << "   dead males: " << deadMales << endl;

}



/* MUTATE GENOTYPE */
void Mutate(Individual &offspring)
{

	// mutate pA
    if (gen > mutGen) // fixed from gen 0 to mutGen, evolves thereafter
    {
    if (Uniform(mt)<mutPa)
      Uniform(mt)<0.5? offspring.Pa0-=Uniform(mt)*stepPa : offspring.Pa0+=Uniform(mt)*stepPa;
    if (Uniform(mt)<mutPa)
      Uniform(mt)<0.5? offspring.Pa1-=Uniform(mt)*stepPa : offspring.Pa1+=Uniform(mt)*stepPa;
    }

    // mutate pB
    if (gen > mutGen) // fixed from gen 0 to mutGen, evolves thereafter
    {
    if (Uniform(mt)<mutPb)
      Uniform(mt)<0.5? offspring.Pb0-=Uniform(mt)*stepPb : offspring.Pb0+=Uniform(mt)*stepPb;
    if (Uniform(mt)<mutPb)
      Uniform(mt)<0.5? offspring.Pb1-=Uniform(mt)*stepPb : offspring.Pb1+=Uniform(mt)*stepPb;
    }

    // mutate pC
    if (gen > mutGen) // fixed from gen 0 to mutGen, evolves thereafter
    {
    if (Uniform(mt)<mutPc)
      Uniform(mt)<0.5? offspring.Pc0-=Uniform(mt)*stepPc : offspring.Pc0+=Uniform(mt)*stepPc;
    if (Uniform(mt)<mutPc)
      Uniform(mt)<0.5? offspring.Pc1-=Uniform(mt)*stepPc : offspring.Pc1+=Uniform(mt)*stepPc;
    }

    //mutate Alpha
    if (Uniform(mt)<mutAlpha)
      Uniform(mt)<0.5? offspring.A0-=Uniform(mt)*stepAlpha : offspring.A0+=Uniform(mt)*stepAlpha;
    if (Uniform(mt)<mutAlpha)
      Uniform(mt)<0.5? offspring.A1-=Uniform(mt)*stepAlpha : offspring.A1+=Uniform(mt)*stepAlpha;

    //mutate Beta
    if (Uniform(mt)<mutBeta)
      Uniform(mt)<0.5? offspring.B0-=Uniform(mt)*stepBeta : offspring.B0+=Uniform(mt)*stepBeta;
    if (Uniform(mt)<mutBeta)
      Uniform(mt)<0.5? offspring.B1-=Uniform(mt)*stepBeta : offspring.B1+=Uniform(mt)*stepBeta;

    // ensure that Beta is not hidden from selection
    if ((offspring.A0+offspring.A1<0.0) && (offspring.B0+offspring.B1<0.0))
    {
        offspring.B0=0.0;
        offspring.B1=0.0;
    }

    //mutate Gamma
    if (Uniform(mt)<mutGamma)
      Uniform(mt)<biasGamma? offspring.G0-=Uniform(mt)*stepGamma : offspring.G0+=Uniform(mt)*stepGamma;
    if (Uniform(mt)<mutGamma)
      Uniform(mt)<biasGamma? offspring.G1-=Uniform(mt)*stepGamma : offspring.G1+=Uniform(mt)*stepGamma;
    if (offspring.G0<0.0) offspring.G0=0.0; // prevent going negative
    if (offspring.G1<0.0) offspring.G1=0.0; // prevent going negative

    //mutate C
    if (Uniform(mt)<mutC)
      {
        if (offspring.C==0) {Uniform(mt)<0.5? offspring.C=1 : offspring.C=2;}
        else if (offspring.C==1) {Uniform(mt)<0.5? offspring.C=0 : offspring.C=2;}
        else if (offspring.C==2) {Uniform(mt)<0.5? offspring.C=0 : offspring.C=1;};
      }

}




/* FEMALE CHOOSES MATE */
void Choose(double pA,double pB,double pC,  double fec, int &Father, int SampleSize)
{
	int i,j, Candidates[SampleSize];
	double  DisplayFix[SampleSize], DisplayFlex[SampleSize], Attractiveness[SampleSize], sumAttractiveness, r;

    uniform_int_distribution<int> RandomMale(0, NM-1); // randomly selected male

 	// select random males
	i=0;
	while (i<SampleSize)
    {
       Candidates[i]=RandomMale(mt); // candidate male randomly selected from adult population
       if (M0[Candidates[i]].alive == 1) i++; // providing male is not dead, include him in sample and move on to next one
    }

	//males choose their level of courtship effort
    sumAttractiveness = 0.0;
    for (i=0;i<SampleSize;i++)
    {
       if ( (M0[Candidates[i]].C == 0) || (M0[Candidates[i]].C == 1 && fec > meanFec) || (M0[Candidates[i]].C == 2 && fec <= meanFec))
       {
       // FIXED PART OF DISPLAY
       DisplayFix[i] = M0[Candidates[i]].t; // fixed display

       // FLEXIBLE PART OF DISPLAY

       // intercept term: display level when fecundity is zero
       DisplayFlex[i] = M0[Candidates[i]].alpha;
       // slope term: can increase (+ve alpha) or decrease (-ve alpha) display level as a power function of female's fecundity
       if (M0[Candidates[i]].beta == 0.0) DisplayFlex[i] += 0.0; // display unaffected by female fecundity
       if (M0[Candidates[i]].beta > 0.0) DisplayFlex[i] += pow(fec,M0[Candidates[i]].beta); // display increases with female fecundity
       if (M0[Candidates[i]].beta < 0.0) DisplayFlex[i] -= pow(fec,-M0[Candidates[i]].beta); // display decreases with female fecundity

       if(DisplayFix[i] < 0.0) DisplayFix[i]=0.0; // minimum trait level is zero
       if(DisplayFlex[i] < 0.0) DisplayFlex[i]=0.0; // minimum display level is zero

       if (killedMales < NM-survdisplay && Uniform(mt)<1-exp(-c*DisplayFlex[i])) { M0[Candidates[i]].alive=0; killedMales++; } // up to XX% of males are killed while displaying
       }

       else DisplayFix[i]=0.0, DisplayFlex[i]=0.0; // male chooses not to display

       if (M0[Candidates[i]].alive==1)
         { Attractiveness[i] = exp(a1*pA*DisplayFix[i] + a2*pB*DisplayFlex[i] + a1*a2*pC*DisplayFix[i]*DisplayFlex[i]);
         sumAttractiveness += Attractiveness[i]; } // male's attractiveness to this particular female
         else { sumAttractiveness += 0.0; } // zero chance of being picked if dead
    }

    Father=9999; // default if all males in sample are dead
    r = Uniform(mt);
    for (i=0;i<SampleSize;i++)
    {
      if ( (r*sumAttractiveness <= Attractiveness[i])&& M0[Candidates[i]].alive==1 )
      {
        Father=Candidates[i]; // choose male who is alive and has most attractive display
        break;
      }
    }
}



/* PRODUCE OFFSPRING */
void CreateKid(int Mother, int Father, Individual &Kid)
{
 	Uniform(mt)<0.5? Kid.Pa0=F0[Mother].Pa0 : Kid.Pa0=F0[Mother].Pa1;
 	Uniform(mt)<0.5? Kid.Pa1=M0[Father].Pa0 : Kid.Pa1=M0[Father].Pa1;
 	Uniform(mt)<0.5? Kid.Pb0=F0[Mother].Pb0 : Kid.Pb0=F0[Mother].Pb1;
 	Uniform(mt)<0.5? Kid.Pb1=M0[Father].Pb0 : Kid.Pb1=M0[Father].Pb1;
 	Uniform(mt)<0.5? Kid.Pc0=F0[Mother].Pc0 : Kid.Pc0=F0[Mother].Pc1;
 	Uniform(mt)<0.5? Kid.Pc1=M0[Father].Pc0 : Kid.Pc1=M0[Father].Pc1;
 	Uniform(mt)<0.5? Kid.A0=F0[Mother].A0 : Kid.A0=F0[Mother].A1;
 	Uniform(mt)<0.5? Kid.A1=M0[Father].A0 : Kid.A1=M0[Father].A1;
 	Uniform(mt)<0.5? Kid.B0=F0[Mother].B0 : Kid.B0=F0[Mother].B1;
 	Uniform(mt)<0.5? Kid.B1=M0[Father].B0 : Kid.B1=M0[Father].B1;
 	Uniform(mt)<0.5? Kid.G0=F0[Mother].G0 : Kid.G0=F0[Mother].G1;
 	Uniform(mt)<0.5? Kid.G1=M0[Father].G0 : Kid.G1=M0[Father].G1;
 	Uniform(mt)<0.5? Kid.V0=F0[Mother].V0 : Kid.V0=F0[Mother].V1;
 	Uniform(mt)<0.5? Kid.V1=M0[Father].V0 : Kid.V1=M0[Father].V1;
 	Kid.C=M0[Father].C;
}



/* PAIRING AND REPRODUCTION TO PRODUCE NEXT GENERATION */
void NextGen()
{
	int fem, off, Mother, Father, NumSons=0, NumDaughters=0;
	double m,pA,pB,pC,fec;
	Individual Kid;

    // pair females with males
	fem = 0;
	while(fem<NF)
	{
      if (F0[fem].alive==1)
      {
        pA=F0[fem].pA; // female's preference
        pB=F0[fem].pB; // female's preference
        pC=F0[fem].pC; // female's preference for interaction term
        fec=F0[fem].f; // female's fecundity
        SampleSize=Poisson(mt)+1; // number of males displaying
        Choose(pA,pB,pC,fec,Father,SampleSize);
        if (Father != 9999) { F0[fem].partner = Father; fem++; }
      }
      else fem++;
	}

	uniform_int_distribution<int> RandomFemale(0, NF-1); // males picked from the distribution

    off= 0; // offspring index
    while (off<N)
	{
      // select mother with chance proportional to fecundity
      m = Uniform(mt);
      for (fem=0;fem<NF;fem++)
      {
        if ( m <= F0[fem].fCum )
        {
          Mother=fem;
          break;
        }
      }

      // produce offspring
      if ( F0[Mother].alive == 1 ) // check that mother is alive
      {
        // populate offspring generation
        Father = F0[Mother].partner;
	    CreateKid(Mother,Father,Kid);
        M0[Father].nKids++;
        Mutate(Kid);

        //determine sex
        if(Uniform(mt) < 0.5)
	    { M1[NumSons]=Kid; NumSons++; }
	    else
        { F1[NumDaughters]=Kid; NumDaughters++; }

        off++;
      }
    }

 //   cout << "gen: " << gen << "   killed males: " << killedMales << endl;

	NM=NumSons; NF=NumDaughters;
	for (off=0;off<NM;off++) M0[off]=M1[off]; //overwrite parental generations with offspring (male)
	for (off=0;off<NF;off++) F0[off]=F1[off]; //overwrite parental generations with offspring (female)

//    cout << "killedMales = " << killedMales << endl;
    killedMales = 0;
}



/* CALCULATE STATISTICS */
void Statistics()
{
    int i,C;
	double pA,pB,pC,v,alpha,beta,gamma,sumPa=0.0,sumPb=0.0,sumPc=0.0,sumV=0.0,sumAlpha=0.0,sumBeta=0.0,sumGamma=0.0, sumAlpha1=0.0,sumBeta1=0.0,sumGamma1=0.0,sumAlpha2=0.0,sumBeta2=0.0,sumGamma2=0.0,sumC0=0.0, sumC1=0.0, sumC2=0.0,
        sumsqPa=0.0,sumsqPb=0.0,sumsqPc=0.0,sumsqV=0.0,sumsqAlpha=0.0,sumsqBeta=0.0,sumsqGamma=0.0,sumprodVAlpha=0.0,sumprodVBeta=0.0,sumprodVGamma=0.0;
	double varPa,varPb,varPc,varV,varAlpha,varBeta,varGamma;

	for (i=0;i<NF;i++)
	{
		pA=F0[i].pA; // preference value
		pB=F0[i].pB; // preference value
		pC=F0[i].pC;
		v=F0[i].vb; // genetic viability
		alpha=0.5*(F0[i].A0 + F0[i].A1);
		beta=0.5*(F0[i].B0 + F0[i].B1);
		gamma=0.5*(F0[i].G0 + F0[i].G1);

		sumPa+=pA;
		sumsqPa+=pA*pA;

		sumPb+=pB;
		sumsqPb+=pB*pB;

		sumPc+=pC;
		sumsqPc+=pC*pC;

		sumV+=v;
		sumsqV+=v*v;

        sumAlpha+=alpha;
		sumsqAlpha+=alpha*alpha;

		sumBeta+=beta;
		sumsqBeta+=beta*beta;

		sumGamma+=gamma;
		sumsqGamma+=gamma*gamma;

		if(C==1)
        {
        sumAlpha1+=alpha;
		sumBeta1+=beta;
		sumGamma1+=gamma;
		}

        if(C==2)
        {
        sumAlpha2+=alpha;
		sumBeta2+=beta;
		sumGamma2+=gamma;
		}


		sumprodVAlpha+=v*alpha;
		sumprodVBeta+=v*beta;
		sumprodVGamma+=v*gamma;

   	}

	for (i=0;i<NM;i++)
	{
        if(M0[i].C==0){sumC0++;}
        if(M0[i].C==1){sumC1++;}
        if(M0[i].C==2){sumC2++;}
	}

	meanPa=sumPa/double(NF);
	meanPb=sumPb/double(NF);
	meanPc=sumPc/double(NF);
	meanV=sumV/double(NF);
    meanAlpha=sumAlpha/double(NF);
    meanBeta=sumBeta/double(NF);
    meanGamma=sumGamma/double(NF);
  	C0=sumC0/double(NM);
   	C1=sumC1/double(NM);
   	C2=sumC2/double(NM);

    meanAlpha1=sumAlpha1/double(sumC1);
    meanBeta1=sumBeta1/double(sumC1);
    meanGamma1=sumGamma1/double(sumC1);

    meanAlpha2=sumAlpha2/double(sumC2);
    meanBeta2=sumBeta2/double(sumC2);
    meanGamma2=sumGamma2/double(sumC2);

	varPa=sumsqPa/double(NF)-meanPa*meanPa;
	varPb=sumsqPb/double(NF)-meanPb*meanPb;
	varPc=sumsqPc/double(NF)-meanPc*meanPc;
	varV=sumsqV/double(NF)-meanV*meanV;
	varAlpha=sumsqAlpha/double(NF)-meanAlpha*meanAlpha;
	varBeta=sumsqBeta/double(NF)-meanBeta*meanBeta;
	varGamma=sumsqGamma/double(NF)-meanGamma*meanGamma;

// to know if there is a problem, variability cannot be negative
	varPa>0? stdevPa=sqrt(varPa):stdevPa=0;
	varPb>0? stdevPb=sqrt(varPb):stdevPb=0;
	varPc>0? stdevPc=sqrt(varPc):stdevPc=0;
	varV>0? stdevV=sqrt(varV):stdevV=0;
	varAlpha>0? stdevAlpha=sqrt(varAlpha):stdevAlpha=0;
	varBeta>0? stdevBeta=sqrt(varBeta):stdevBeta=0;
	varGamma>0? stdevGamma=sqrt(varGamma):stdevGamma=0;

	(stdevV>0 && stdevAlpha>0)? corr_vAlpha=(sumprodVAlpha/double(NF)-meanV*meanAlpha)/(stdevV*stdevAlpha):corr_vAlpha=0;
	(stdevV>0 && stdevBeta>0)? corr_vBeta=(sumprodVBeta/double(NF)-meanV*meanBeta)/(stdevV*stdevBeta):corr_vBeta=0;
	(stdevV>0 && stdevGamma>0)? corr_vGamma=(sumprodVGamma/double(NF)-meanV*meanGamma)/(stdevV*stdevGamma):corr_vGamma=0;

}



/* WRITE PARAMETER SETTINGS TO OUTPUT FILE */
void printparams()
{
  fout << endl << "PARAMETER VALUES" << endl
       << "mutPa: " << "\t" << setprecision(4) << mutPa << endl
       << "stepPa: " << "\t" << setprecision(4) << stepPa << endl
       << "mutPb: " << "\t" << setprecision(4) << mutPb << endl
       << "stepPb: " << "\t" << setprecision(4) << stepPb << endl
       << "mutPc: " << "\t" << setprecision(4) << mutPc << endl
       << "stepPc: " << "\t" << setprecision(4) << stepPc << endl
       << "mutV: " << "\t" << setprecision(4) << mutV << endl
       << "stepV: " << "\t" << setprecision(4) << stepV << endl
       << "mutAlpha: " << "\t" << setprecision(4) << mutAlpha << endl
       << "stepAlpha: " << "\t" << setprecision(4) << stepAlpha << endl
       << "mutBeta: " << "\t" << setprecision(4) << mutBeta << endl
       << "stepBeta: " << "\t" << setprecision(4) << stepBeta << endl
       << "mutGamma: " << "\t" << setprecision(4) << mutGamma << endl
       << "stepGamma: " << "\t" << setprecision(4) << stepGamma << endl
       << "biasGamma: " << "\t" << setprecision(4) << biasGamma << endl
       << "a1: " << "\t" << a1 << endl
       << "a2: " << "\t" << a2 << endl
       << "b: " << "\t" << b << endl
       << "c: " << "\t" << c << endl
       << "k: " << "\t" << k << endl
       << "Vopt: " << "\t" << setprecision(4) << Vopt << endl
       << "avNumComp: " << "\t" << setprecision(4) << avNumComp << endl
       << "survdisplay: " << "\t" << setprecision(4) << survdisplay << endl
       << "initPa: " << "\t" << setprecision(4) << initPa << endl
       << "initPb: " << "\t" << setprecision(4) << initPb << endl
       << "initPc: " << "\t" << setprecision(4) << initPc << endl
       << "initV: " << "\t" << setprecision(4) << initV << endl
       << "initAlpha: " << "\t" << setprecision(4) << initAlpha << endl
       << "initBeta: " << "\t" << setprecision(4) << initBeta << endl
       << "initGamma: " << "\t" << setprecision(4) << initBeta << endl
       << "NumGen: " << "\t" << NumGen << endl;
}



/* WRITE GENETIC TRAIT VALUES TO OUTPUT FILE */
void WriteMeans()
{

  // show values on screen
  cout << setw(6) << gen
       << setw(9) << setprecision(4) << meanPa
       << setw(9) << setprecision(4) << meanPb
       << setw(9) << setprecision(4) << meanPc
	   << setw(9) << setprecision(4) << meanV
	   << setw(9) << setprecision(4) << meanAlpha
	   << setw(9) << setprecision(4) << meanBeta
       /*<< setw(9) << setprecision(4) << meanAlpha1
	   << setw(9) << setprecision(4) << meanBeta1
	   << setw(9) << setprecision(4) << meanAlpha2
	   << setw(9) << setprecision(4) << meanBeta2*/
	   << setw(9) << setprecision(4) << meanGamma
	   << setw(9) << setprecision(4) << C0
	   << setw(9) << setprecision(4) << C1
	   << setw(9) << setprecision(4) << C2
       << setw(9) << NM
       << setw(9) << NF
	   << endl;


  // write values to output file
  fout << gen
       << "\t" << setprecision(4) << meanPa
       << "\t" << setprecision(4) << meanPb
       << "\t" << setprecision(4) << meanPc
	   << "\t" << setprecision(4) << meanV
	   << "\t" << setprecision(4) << meanAlpha
	   << "\t" << setprecision(4) << meanBeta
	   << "\t" << setprecision(4) << meanGamma
	   << "\t" << setprecision(4) << stdevPa
	   << "\t" << setprecision(4) << stdevPb
	   << "\t" << setprecision(4) << stdevPc
	   << "\t" << setprecision(4) << stdevV
	   << "\t" << setprecision(4) << stdevAlpha
	   << "\t" << setprecision(4) << stdevBeta
	   << "\t" << setprecision(4) << stdevGamma
	   << "\t" << setprecision(4) << C0
	   << "\t" << setprecision(4) << C1
	   << "\t" << setprecision(4) << C2
	   << "\t" << setprecision(4) << corr_vAlpha
	   << "\t" << setprecision(4) << corr_vBeta
	   << "\t" << setprecision(4) << corr_vGamma
	   << "\t" << NM
	   << endl;
}



/* MAIN PROGRAM */
int main()
{
    for(rep=0;rep<numrep;rep++)
    {

    gen=0; // generation zero
    cout <<  setw(6) << "gen" << setw(9) << "pA" << setw(9) << "pB" << setw(9) << "pC" << setw(9)<< "v" << setw(9) << "alpha" << setw(9) << "beta" << setw(9) /*<< "alpha1" << setw(9) << "beta1" << setw(9)<< "alpha2" << setw(9) << "beta2" << setw(9)*/<< "gamma"
    << setw(9)<< "C0" << setw(9)<< "C1" << setw(9) << "C2" << setw(9) << "NM" << setw(9) << "NF" << endl; // column headings on screen
	fout << "generation" << "\t" << "meanPa" << "\t" << "meanPb" << "\t"<< "meanPc"<< "\t"<< "meanV" << "\t" << "meanAlpha" << "\t" << "meanBeta" << "\t" << "meanGamma" << "\t"
         << "stdevPa" << "\t" << "stdevPb" << "\t"<< "stdevPc" << "\t" << "stdevV" << "\t" << "stdevAlpha" << "\t" << "stdevBeta" << "\t" << "stdevGamma" << "\t"
         << "C0" << "\t" << "C1" << "\t" << "C2" << "\t" << "corr_vAlpha" << "\t"  << "corr_vBeta" << "\t" << "corr_vGamma" << "\t" << "NM" << endl; // column headings in output file
	Init();
	Phenotypes();
	Statistics();
	WriteMeans();

	for (gen=1;gen<=NumGen;gen++)
	{
	      NextGen();
          Phenotypes();
		  if (gen%skip==0){ Statistics(); WriteMeans();} // write output every 'skip' generations
	}

    fout << endl << endl << endl;

    }
    printparams();

	return 0;
}
