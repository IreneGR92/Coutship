/***********************************************
 Good-genes model of sexual selection on male courtship effort, based on 'revealing handicap' mechanism as described by van Doorn & Weissing (2006).
 Males with high genetic viability enjoy greater marginal fecundity benefits of displaying than do males with low genetic viability.
 Female has evolvable preferences for fixed and flexible display elements. Male can adjust level of flexible display depending on female's fecundity.
 July 2017, Exeter
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
#include <vector>
#include <algorithm>
#include <chrono>


// constants, type definitions, etc.

using namespace std;

const int seed         = time(0); // pseudo-random seed
// const int seed         = <<enter seed here>>;

const int N            = 1000;    // population size

const double mutPfix   = 0.05;    // mutation rate in p for the fixed display
const double stepPfix  = 0.4;     // mutation step size in p for the fixed display
const double mutPflex  = 0.05;    // mutation rate in p for the flexible display
const double stepPflex = 0.4;     // mutation step size in p for the flexible display
const double mutPint   = 0.00;    // mutation rate in p for the interaction
const double stepPint  = 0.4;     // mutation step size in p for the interaction
const double mutAlpha  = 0.0;    // mutation rate in alpha
const double stepAlpha = 0.4;     // mutation step size in alpha
const double mutBeta   = 0.0;    // mutation rate in beta
const double stepBeta  = 0.4;     // mutation step size in beta
const double mutTrait  = 0.05;    // mutation rate in trait
const double stepTrait = 0.4;     // mutation step size in trait
const double mutV      = 0.05;    // mutation rate in v
const double stepV     = 0.8;     // mutation step size in v
const double biasV     = 0.9;     // prob that mutation in v is downwards (0.5 = unbiased)

const double a1        = 1.0;     // strength of preference for the display
const double a2        = 1.0;     // strength of preference for flexible display
const double a3        = a1*a2;   // strength of preference for interaction between fixed and flexible displays
const double b         = 0.001;   // cost of each preference component
const double k         = 1.0;     // cost of fixed display
const double c         = 0.1;     // cost of flexible display
const double Vopt      = 10.0;    // optimal value of v

const double avNumComp = 5.0;     // number of competing males is drawn from Y~Poisson(avNumComp)

const double initPfix  = 1.0;     // starting value of female preference for fixed display (in gen 0)
const double initPflex = 1.0;     // starting value of female preference for flexible display (in gen 0)
const double initPint  = 0.0;     // starting value of female preference for interaction between displays (in gen 0)
const double initAlpha = 0.0;     // starting value of alpha (in gen 0)
const double initBeta  = 0.0;     // starting value of beta (in gen 0)
const double initTrait = 0.0;     // starting value of (investment in) trait (in gen 0)
const double initV     = Vopt;    // starting value of v (in gen 0)
const double meanFec   = 5.0;     // mean female fecundity
const double stdevFec  = 1.0;     // standard deviation of mean female fecundity

const int NumGen       = 10000;   // number of generations
const int mutGen       = 2000;    // number of generations after which p mutates
const int propkilled   = 10;      // after flexible display --> killedMales < NM/propkilled
const int numrep       = 20;      // number of replicates
const int skip         = 50;      // interval between print-outs

struct Individual // define individual traits
{
  double Pfix0,Pfix1,Pflex0,Pflex1,Pint0,Pint1,A0,A1,B0,B1,T0,T1,V0,V1, // genetic values for female preferences, male strategy and viability
    pFix,pFlex,pInt,alpha,beta,t,v,viab,f,VF;  // phenotypic values
  int nKids;              // number of offspring
};

typedef Individual Population[N];

Population F0,F1,M0,M1;           // female and male parent and offspring generations
int NF,NM,SampleSize,gen,rep,killedMales;    // number of females, males and number of males sampled by female
double sumViabFec,sumViab,
  meanPfix,meanPflex,meanPint,meanV,
  meanAlpha,meanBeta,meanTrait,
  stdevPfix,stdevPflex,stdevPint,stdevV,
  stdevAlpha,stdevBeta,stdevTrait,
  corr_vAlpha,corr_vBeta,corr_vTrait;         // population means, standard deviations and correlations

ofstream fout("display.txt");     // output file
ofstream hout("census.txt");      // second output file with snapshot census of male display elements and genetic quality

// random numbers
mt19937 mt(seed); // random number generator
uniform_real_distribution<double> Uniform(0, 1); // real number between 0 and 1 (uniform)
poisson_distribution<int> Poisson(avNumComp); // random draw from Poisson distribution with mean avNumComp


/* INITIALISE POPULATION */
void Init()
{
	int i;

	for (i=0;i<N/2;i++) // for half of individuals in population
	  {
      F0[i].Pfix0=initPfix;
      F0[i].Pfix1=initPfix;
      F0[i].Pflex0=initPflex;
      F0[i].Pflex1=initPflex;
      F0[i].Pint0=initPint;
      F0[i].Pint1=initPint;
      F0[i].A0=initAlpha;
	  F0[i].A1=initAlpha;
	  F0[i].B0=initBeta;
	  F0[i].B1=initBeta;
	  F0[i].T0=initTrait;
	  F0[i].T1=initTrait;
	  F0[i].V0=initV;
	  F0[i].V1=initV;

      M0[i].Pfix0=initPfix;
      M0[i].Pfix1=initPfix;
      M0[i].Pflex0=initPflex;
      M0[i].Pflex1=initPflex;
      M0[i].Pint0=initPint;
      M0[i].Pint1=initPint;
      M0[i].A0=initAlpha;
      M0[i].A1=initAlpha;
      M0[i].B0=initBeta;
      M0[i].B1=initBeta;
	  M0[i].T0=initTrait;
	  M0[i].T1=initTrait;
	  M0[i].V0=initV;
	  M0[i].V1=initV;
      }
	NF=NM=N/2; // females and males equally abundant in starting population
}



/* CALCULATE PHENOTYPES AND RESCALE FITNESS */
void Phenotypes()
{
	int i, deadMales = 0, deadFemales = 0;

	sumViabFec=0.0;
	sumViab=0.0;

	// Calculate female trait values and absolute viabilities
	for (i=0;i<NF;i++) // for all females
	{
		F0[i].pFix = 0.5*(F0[i].Pfix0+F0[i].Pfix1); // pref-fixed is average of maternal and paternal genes
		F0[i].pFlex = 0.5*(F0[i].Pflex0+F0[i].Pflex1); // pref-flexible is average of maternal and paternal genes
		F0[i].pInt = 0.5*(F0[i].Pint0+F0[i].Pint1); // pref-interaction is average of maternal and paternal genes
		F0[i].v = 0.5*(F0[i].V0+F0[i].V1); // genetic viability is average of maternal and paternal genes

		F0[i].viab = exp(-b*F0[i].pFix*F0[i].pFix - b*F0[i].pFlex*F0[i].pFlex - b*F0[i].pInt*F0[i].pInt - abs(Vopt - F0[i].v));  // female viability is reduced by expressed pref and distance from Vopt

        normal_distribution<double> Normal(meanFec, stdevFec); // random draw from normal distribution
		F0[i].f = Normal(mt); // female fecundity randomly drawn from normal distribution
		if (F0[i].f < 0.0) F0[i].f = 0.0; // reset to zero if negative

		F0[i].VF = sumViabFec + F0[i].viab*F0[i].f; // female offspring production is proportional to viability*fecundity
		sumViabFec = F0[i].VF; // keeps cumulative total (for later use)

        // Reset beta to 0
        if(F0[i].alpha < 0.0 && F0[i].beta < 0.0) { F0[i].beta=0.0; F0[i].B0=0.0; F0[i].B1=0.0; }  //when selection is neutral (alpha and beta negative, we make beta = 0)
	}

	// Calculate male trait values
	for (i=0;i<NM;i++)
	{
	    M0[i].alpha = 0.5*(M0[i].A0 + M0[i].A1); // intercept of display function (average of maternal and paternal genes)
        M0[i].beta = 0.5*(M0[i].B0 + M0[i].B1); // slope of display function (average of maternal and paternal genes)
        M0[i].t = 0.5*(M0[i].T0 + M0[i].T1); // investment in static trait; average of maternal and paternal genes
		M0[i].v = 0.5*(M0[i].V0 + M0[i].V1); // genetic viability is average of maternal and paternal genes

		M0[i].viab = sumViab + exp(-k*M0[i].t*M0[i].t - abs(Vopt - M0[i].v)); // male viability depends on expressed trait and distance from Vopt
		sumViab = M0[i].viab; // keeps cumulative total (for later use)

        M0[i].nKids = 0;

        // Reset beta to 0
        if(M0[i].alpha < 0.0 && M0[i].beta < 0.0) { M0[i].beta=0.0; M0[i].B0=0.0; M0[i].B1=0.0; } //when selection is neutral (alpha and beta negative, we make beta = 0)
	}

}



/* MUTATE GENOTYPE */
void Mutate(Individual &offspring)
{

	// mutate pFix
    if (gen > mutGen) // fixed from gen 0 to mutGen, evolves thereafter
    {
    if (Uniform(mt)<mutPfix)
      Uniform(mt)<0.5? offspring.Pfix0-=Uniform(mt)*stepPfix : offspring.Pfix0+=Uniform(mt)*stepPfix;
    if (Uniform(mt)<mutPfix)
      Uniform(mt)<0.5? offspring.Pfix1-=Uniform(mt)*stepPfix : offspring.Pfix1+=Uniform(mt)*stepPfix;
    }

    // mutate pFlex
    if (gen > mutGen) // fixed from gen 0 to mutGen, evolves thereafter
    {
    if (Uniform(mt)<mutPflex)
      Uniform(mt)<0.5? offspring.Pflex0-=Uniform(mt)*stepPflex : offspring.Pflex0+=Uniform(mt)*stepPflex;
    if (Uniform(mt)<mutPflex)
      Uniform(mt)<0.5? offspring.Pflex1-=Uniform(mt)*stepPflex : offspring.Pflex1+=Uniform(mt)*stepPflex;
    }

    // mutate pInt
    if (gen > mutGen) // fixed from gen 0 to mutGen, evolves thereafter
    {
    if (Uniform(mt)<mutPint)
      Uniform(mt)<0.5? offspring.Pint0-=Uniform(mt)*stepPint : offspring.Pint0+=Uniform(mt)*stepPint;
    if (Uniform(mt)<mutPint)
      Uniform(mt)<0.5? offspring.Pint1-=Uniform(mt)*stepPint : offspring.Pint1+=Uniform(mt)*stepPint;
    }

	// mutate v
    if (Uniform(mt)<mutV)
      Uniform(mt)<biasV? offspring.V0-=Uniform(mt)*stepV : offspring.V0+=Uniform(mt)*stepV;
    if (Uniform(mt)<mutV)
      Uniform(mt)<biasV? offspring.V1-=Uniform(mt)*stepV : offspring.V1+=Uniform(mt)*stepV;

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

    //mutate Trait
    if (Uniform(mt)<mutTrait)
      Uniform(mt)<0.5? offspring.T0-=Uniform(mt)*stepTrait : offspring.T0+=Uniform(mt)*stepTrait;
    if (Uniform(mt)<mutTrait)
      Uniform(mt)<0.5? offspring.T1-=Uniform(mt)*stepTrait : offspring.T1+=Uniform(mt)*stepTrait;
    if (offspring.T0<0.0) offspring.T0=0.0; // prevent going negative
    if (offspring.T1<0.0) offspring.T1=0.0; // prevent going negative
}




/* FEMALE CHOOSES MATE */
void Choose(double pFix, double pFlex, double pInt, double fec, int &Father, int SampleSize)
{
	int i, j, mal, Candidates[SampleSize];
	double DisplayTotal[SampleSize], DisplayFix[SampleSize], DisplayFlex[SampleSize], Attractiveness[SampleSize], sumAttractiveness, r, s, dropViab;

 	// select random males
	i=0;
	while (i<SampleSize)
    {
      // select males with chance proportional to viability
      s = Uniform(mt)*sumViab;
//cout << "s = " << s << endl;
      for (mal=0;mal<NM;mal++)
      {
        if ( s <= M0[mal].viab )
        {
          Candidates[i]=mal;
          break;
        }
      }
      i++;
    }

	//males choose their level of courtship effort
    sumAttractiveness = 0.0;
    for (i=0;i<SampleSize;i++)
    {
       // FIXED PART OF DISPLAY
       DisplayFix[i] = M0[Candidates[i]].t; // fixed display

       // FLEXIBLE PART OF DISPLAY
       // intercept term: display level when zero competitors
       DisplayFlex[i] = M0[Candidates[i]].alpha;
       // slope term: can increase (+ve alpha) or decrease (-ve alpha) display level as a power function of female's fecundity
       if (M0[Candidates[i]].beta == 0.0) DisplayFlex[i] += 0.0; // display unaffected by female fecundity
       if (M0[Candidates[i]].beta > 0.0) DisplayFlex[i] += pow(fec,M0[Candidates[i]].beta); // display increases with female fecundity
       if (M0[Candidates[i]].beta < 0.0) DisplayFlex[i] -= pow(fec,-M0[Candidates[i]].beta); // display decreases with female fecundity

       if(DisplayFix[i] < 0.0) DisplayFix[i]=0.0; // minimum display level is zero
       if(DisplayFlex[i] < 0.0) DisplayFlex[i]=0.0;

       // revealing indicator model, so multiply each component of display by male's genetic viability (males with higher viability benefit more)
       Attractiveness[i] = sumAttractiveness + exp(a1*pFix*DisplayFix[i]*exp(-abs(Vopt - M0[Candidates[i]].v)) + a2*pFlex*DisplayFlex[i]*exp(-abs(Vopt - M0[Candidates[i]].v)) + a3*pInt*DisplayFix[i]*DisplayFlex[i]*exp(-abs(Vopt - M0[Candidates[i]].v))); // male's attractiveness to this particular female
       sumAttractiveness = Attractiveness[i]; // keeps cumulative total (for later use)

       // calculate reductions in viability for displaying males; only depends on display level (not on genetic viability)
       if (Candidates[i]==0) { dropViab = M0[Candidates[i]].viab*(1-exp(-c*DisplayFlex[i]*DisplayFlex[i])); }
       else { dropViab = ( M0[Candidates[i]].viab - M0[Candidates[i]-1].viab ) * (1-exp(-c*DisplayFlex[i]*DisplayFlex[i])); }
       for (j=Candidates[i];j<NM;j++)
       {
         M0[j].viab-=dropViab; // reduce cumulative viabilities
         sumViab=M0[j].viab; // keeps cumulative total (for later use)
       }
    }


    r = Uniform(mt);
    for (i=0;i<SampleSize;i++)
    {
      if ( r*sumAttractiveness <= Attractiveness[i] )
      {
        Father=Candidates[i]; // choose candidate male with chance proportional to attractiveness of his display
        break;
      }
    }

}



/* PRODUCE OFFSPRING */
void CreateKid(int Mother, int Father, Individual &Kid)
{
 	Uniform(mt)<0.5? Kid.Pfix0=F0[Mother].Pfix0 : Kid.Pfix0=F0[Mother].Pfix1;
 	Uniform(mt)<0.5? Kid.Pfix1=M0[Father].Pfix0 : Kid.Pfix1=M0[Father].Pfix1;
 	Uniform(mt)<0.5? Kid.Pflex0=F0[Mother].Pflex0 : Kid.Pflex0=F0[Mother].Pflex1;
 	Uniform(mt)<0.5? Kid.Pflex1=M0[Father].Pflex0 : Kid.Pflex1=M0[Father].Pflex1;
 	Uniform(mt)<0.5? Kid.Pint0=F0[Mother].Pint0 : Kid.Pint0=F0[Mother].Pint1;
 	Uniform(mt)<0.5? Kid.Pint1=M0[Father].Pint0 : Kid.Pint1=M0[Father].Pint1;
 	Uniform(mt)<0.5? Kid.A0=F0[Mother].A0 : Kid.A0=F0[Mother].A1;
 	Uniform(mt)<0.5? Kid.A1=M0[Father].A0 : Kid.A1=M0[Father].A1;
 	Uniform(mt)<0.5? Kid.B0=F0[Mother].B0 : Kid.B0=F0[Mother].B1;
 	Uniform(mt)<0.5? Kid.B1=M0[Father].B0 : Kid.B1=M0[Father].B1;
 	Uniform(mt)<0.5? Kid.T0=F0[Mother].T0 : Kid.T0=F0[Mother].T1;
 	Uniform(mt)<0.5? Kid.T1=M0[Father].T0 : Kid.T1=M0[Father].T1;
 	Uniform(mt)<0.5? Kid.V0=F0[Mother].V0 : Kid.V0=F0[Mother].V1;
 	Uniform(mt)<0.5? Kid.V1=M0[Father].V0 : Kid.V1=M0[Father].V1;
}



/* PAIRING AND REPRODUCTION TO PRODUCE NEXT GENERATION */
void NextGen()
{
	int fem, off, Mother, Father, NumSons=0, NumDaughters=0;
	double m,pFix,pFlex,pInt,fec;
	Individual Kid;

    off = 0; // offspring index
    while (off<N)
	{
      // select mother with chance proportional to viability*fecundity
      m = Uniform(mt)*sumViabFec;
      for (fem=0;fem<NF;fem++)
      {
        if ( m <= F0[fem].VF )
        {
          Mother=fem;
          break;
        }
      }

      pFix=F0[Mother].pFix; // female's preference
      pFlex=F0[Mother].pFlex; // female's preference
      pInt=F0[Mother].pInt; // female's preference for interaction term
      fec=F0[Mother].f; // female's fecundity

      // pair female with male
      SampleSize=Poisson(mt)+1; // determine number of competing males
      Choose(pFix,pFlex,pInt,fec,Father,SampleSize); // select winner

      // produce offspring
      CreateKid(Mother,Father,Kid);
      M0[Father].nKids++;
      Mutate(Kid);

      // populate offspring generation
      if(Uniform(mt) < 0.5) //determine sex
	    { M1[NumSons]=Kid; NumSons++; }
	  else
        { F1[NumDaughters]=Kid; NumDaughters++; }

      off++;
    }

	NM=NumSons; NF=NumDaughters;
	for (off=0;off<NM;off++) M0[off]=M1[off]; //overwrite parental generations with offspring (male)
	for (off=0;off<NF;off++) F0[off]=F1[off]; //overwrite parental generations with offspring (female)

}



/* CALCULATE STATISTICS */
void Statistics()
{
    int i;
	double pFix,pFlex,pInt,v,alpha,beta,trait,sumPfix=0.0,sumPflex=0.0,sumPint=0.0,sumV=0.0,sumAlpha=0.0,sumBeta=0.0,sumTrait=0.0,
        sumsqPfix=0.0,sumsqPflex=0.0,sumsqPint=0.0,sumsqV=0.0,sumsqAlpha=0.0,sumsqBeta=0.0,sumsqTrait=0.0,sumprodVAlpha=0.0,sumprodVBeta=0.0,sumprodVTrait=0.0;
	double varPfix,varPflex,varPint,varV,varAlpha,varBeta,varTrait;

	for (i=0;i<NF;i++)
	{
		pFix=F0[i].pFix; // preference value
		pFlex=F0[i].pFlex; // preference value
		pInt=F0[i].pInt;
		v=F0[i].v; // genetic viability
		alpha=0.5*(F0[i].A0 + F0[i].A1);
		beta=0.5*(F0[i].B0 + F0[i].B1);
		trait=0.5*(F0[i].T0 + F0[i].T1);

		sumPfix+=pFix;
		sumsqPfix+=pFix*pFix;

		sumPflex+=pFlex;
		sumsqPflex+=pFlex*pFlex;

		sumPint+=pInt;
		sumsqPint+=pInt*pInt;

		sumV+=v;
		sumsqV+=v*v;

        sumAlpha+=alpha;
		sumsqAlpha+=alpha*alpha;

		sumBeta+=beta;
		sumsqBeta+=beta*beta;

		sumTrait+=trait;
		sumsqTrait+=trait*trait;

		sumprodVAlpha+=v*alpha;
		sumprodVBeta+=v*beta;
		sumprodVTrait+=v*trait;
   	}
/*
	for (i=0;i<NM;i++)
	{
    cout << "nKids = " << M0[i].nKids << endl;
	}
*/
	meanPfix=sumPfix/double(NF);
	meanPflex=sumPflex/double(NF);
	meanPint=sumPint/double(NF);
	meanV=sumV/double(NF);
    meanAlpha=sumAlpha/double(NF);
    meanBeta=sumBeta/double(NF);
    meanTrait=sumTrait/double(NF);

	varPfix=sumsqPfix/double(NF)-meanPfix*meanPfix;
	varPflex=sumsqPflex/double(NF)-meanPflex*meanPflex;
	varPint=sumsqPint/double(NF)-meanPint*meanPint;
	varV=sumsqV/double(NF)-meanV*meanV;
	varAlpha=sumsqAlpha/double(NF)-meanAlpha*meanAlpha;
	varBeta=sumsqBeta/double(NF)-meanBeta*meanBeta;
	varTrait=sumsqTrait/double(NF)-meanTrait*meanTrait;

// to know if there is a problem, variability cannot be negative
	varPfix>0? stdevPfix=sqrt(varPfix):stdevPfix=0;
	varPflex>0? stdevPflex=sqrt(varPflex):stdevPflex=0;
	varPint>0? stdevPint=sqrt(varPint):stdevPint=0;
	varV>0? stdevV=sqrt(varV):stdevV=0;
	varAlpha>0? stdevAlpha=sqrt(varAlpha):stdevAlpha=0;
	varBeta>0? stdevBeta=sqrt(varBeta):stdevBeta=0;
	varTrait>0? stdevTrait=sqrt(varTrait):stdevTrait=0;

	(stdevV>0 && stdevAlpha>0)? corr_vAlpha=(sumprodVAlpha/double(NF)-meanV*meanAlpha)/(stdevV*stdevAlpha):corr_vAlpha=0;
	(stdevV>0 && stdevBeta>0)? corr_vBeta=(sumprodVBeta/double(NF)-meanV*meanBeta)/(stdevV*stdevBeta):corr_vBeta=0;
	(stdevV>0 && stdevTrait>0)? corr_vTrait=(sumprodVTrait/double(NF)-meanV*meanTrait)/(stdevV*stdevTrait):corr_vTrait=0;


}



/* WRITE PARAMETER SETTINGS TO OUTPUT FILE */
void printparams()
{
  fout << endl << "PARAMETER VALUES FECUNDITY" << endl
       << "mutPfix: " << "\t" << setprecision(4) << mutPfix << endl
       << "stepPfix: " << "\t" << setprecision(4) << stepPfix << endl
       << "mutPflex: " << "\t" << setprecision(4) << mutPflex << endl
       << "stepPflex: " << "\t" << setprecision(4) << stepPflex << endl
       << "mutPint: " << "\t" << setprecision(4) << mutPint << endl
       << "stepPint: " << "\t" << setprecision(4) << stepPint << endl
       << "mutV: " << "\t" << setprecision(4) << mutV << endl
       << "stepV: " << "\t" << setprecision(4) << stepV << endl
       << "mutAlpha: " << "\t" << setprecision(4) << mutAlpha << endl
       << "stepAlpha: " << "\t" << setprecision(4) << stepAlpha << endl
       << "mutBeta: " << "\t" << setprecision(4) << mutBeta << endl
       << "stepBeta: " << "\t" << setprecision(4) << stepBeta << endl
       << "mutTrait: " << "\t" << setprecision(4) << mutTrait << endl
       << "stepTrait: " << "\t" << setprecision(4) << stepTrait << endl
       << "biasV: " << "\t" << setprecision(4) << biasV << endl
       << "a1: " << "\t" << a1 << endl
       << "a2: " << "\t" << a2 << endl
       << "b: " << "\t" << b << endl
       << "c: " << "\t" << c << endl
       << "k: " << "\t" << k << endl
       << "Vopt: " << "\t" << setprecision(4) << Vopt << endl
       << "avNumComp: " << "\t" << setprecision(4) << avNumComp << endl
       << "propkilled: " << "\t" << setprecision(4) << propkilled << endl
       << "initPfix: " << "\t" << setprecision(4) << initPfix << endl
       << "initPflex: " << "\t" << setprecision(4) << initPflex << endl
       << "initPint: " << "\t" << setprecision(4) << initPint << endl
       << "initV: " << "\t" << setprecision(4) << initV << endl
       << "initAlpha: " << "\t" << setprecision(4) << initAlpha << endl
       << "initBeta: " << "\t" << setprecision(4) << initBeta << endl
       << "initTrait: " << "\t" << setprecision(4) << initBeta << endl
       << "NumGen: " << "\t" << NumGen << endl;
}



/* WRITE GENETIC TRAIT VALUES TO OUTPUT FILE */
void WriteMeans()
{

  // show values on screen
  cout << setw(6) << gen
       << setw(9) << setprecision(4) << meanPfix
       << setw(9) << setprecision(4) << meanPflex
       << setw(9) << setprecision(4) << meanPint
	   << setw(9) << setprecision(4) << meanV
	   << setw(9) << setprecision(4) << meanAlpha
	   << setw(9) << setprecision(4) << meanBeta
	   << setw(9) << setprecision(4) << meanTrait
       << setw(9) << setprecision(4) << double(NM)/double(N)
	   << endl;


  // write values to output file
  fout << gen
       << "\t" << setprecision(4) << meanPfix
       << "\t" << setprecision(4) << meanPflex
       << "\t" << setprecision(4) << meanPint
	   << "\t" << setprecision(4) << meanV
	   << "\t" << setprecision(4) << meanAlpha
	   << "\t" << setprecision(4) << meanBeta
	   << "\t" << setprecision(4) << meanTrait
	   << "\t" << setprecision(4) << stdevPfix
	   << "\t" << setprecision(4) << stdevPflex
	   << "\t" << setprecision(4) << stdevPint
	   << "\t" << setprecision(4) << stdevV
	   << "\t" << setprecision(4) << stdevAlpha
	   << "\t" << setprecision(4) << stdevBeta
	   << "\t" << setprecision(4) << stdevTrait
	   << "\t" << setprecision(4) << corr_vAlpha
	   << "\t" << setprecision(4) << corr_vBeta
	   << "\t" << setprecision(4) << corr_vTrait
	   << "\t" << setprecision(4) << double(NM)/double(N)
	   << endl;
}



/* TAKE SNAPSHOT OF MALE VALUES FOR DISPLAY ELEMENTS AND GENETIC QUALITY */
void Census()
{

    int i;

  // write values to output file
	hout << "replicate" << "\t" << "generation" << "\t" << "meanPfix" << "\t" << "meanPflex" << "\t"<< "meanPint" << "\t" << "male" << "\t" << "v" << "\t" << "t" << "\t" << "alpha" << "\t" << "beta" << endl; // column headings in output file
	for (i=0;i<NM;i++)
    {
       hout << rep
       << "\t" << gen
       << "\t" << setprecision(4) << meanPfix
       << "\t" << setprecision(4) << meanPflex
       << "\t" << setprecision(4) << meanPint
       << "\t" << i
	   << "\t" << setprecision(4) << M0[i].v
	   << "\t" << setprecision(4) << M0[i].t
	   << "\t" << setprecision(4) << M0[i].alpha
	   << "\t" << setprecision(4) << M0[i].beta
	   << endl;
    }
    hout << endl << endl << endl;
}



/* MAIN PROGRAM */
int main()
{

    fout << "Random seed: " << seed << endl; // write seed to output file

    for(rep=0;rep<numrep;rep++)
    {

    gen=0; // generation zero
    cout <<  setw(6) << "gen" << setw(9) << "pFix" << setw(9) << "pFlex" << setw(9) << "pInt" << setw(9)<< "v" << setw(9) << "alpha" << setw(9) << "beta" << setw(9) << "trait" << setw(9) << "propM" << setw(9) << endl; // column headings on screen
	fout << "generation" << "\t" << "meanPfix" << "\t" << "meanPflex" << "\t"<< "meanPint"<< "\t"<< "meanV" << "\t" << "meanAlpha" << "\t" << "meanBeta" << "\t" << "meanTrait" << "\t"
         << "stdevPfix" << "\t" << "stdevPflex" << "\t"<< "stdevPint" << "\t" << "stdevV" << "\t" << "stdevAlpha" << "\t" << "stdevBeta" << "\t" << "stdevTrait" << "\t" << "corr_vAlpha" << "\t"  << "corr_vBeta" << "\t" << "corr_vTrait" << "\t" << "NM" << endl; // column headings in output file
	Init();
	Phenotypes();
	Statistics();
	WriteMeans();

	for (gen=1;gen<=NumGen;gen++)
	{
	      NextGen();
          Phenotypes();
		  if (gen%skip==0){ Statistics(); WriteMeans();} // write output every 'skip' generations
		  if (gen==mutGen) { Census(); }
		  if (gen==NumGen) { Census(); }
	}

    fout << endl << endl << endl;

    }
    printparams();

	return 0;
}
