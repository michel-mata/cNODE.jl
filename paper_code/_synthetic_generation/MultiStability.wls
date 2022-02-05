#!/usr/bin/env wolframscript
(* ::Package:: *)

scriptPath = ExpandFileName[First[$ScriptCommandLine]];
scriptName = Last[FileNameSplit[ scriptPath ]];

Print[scriptPath];
Print[scriptName];


x = Table[ToExpression["x"<>ToString[i]],{i,1,100}];

Print["Ready"];
AbsoluteTiming[Table[
	Z=Import["./Data/Synthetic/100/Strength/Z0_"<>ToString@j<>".csv"];
	P=Import["./Data/Synthetic/100/Strength/P0_"<>ToString@j<>".csv"];
	Xa={};Xb={};
	h=Min/@(Transpose@Table[1/(4 v^2/.x_/;x==0->1),{v,Transpose[P]}]/.x_/;x==1->Infinity);
	Table[
		eqns=Thread[v*h*x^2-x+v==0];
		vals=Table[NSolve[eqns[[k]],x[[k]],Reals],{k,1,100}];
		AppendTo[Xa,x/.Flatten[First/@vals]];
		AppendTo[Xb,x/.Flatten[Last/@vals]];,
	{v,Transpose[P]}];
	Export["./Data/Synthetic/100/MultiStability/H"<>ToString@j<>".csv",h];
	Export["./Data/Synthetic/100/MultiStability/Z"<>ToString@j<>".csv",Z];
	Export["./Data/Synthetic/100/MultiStability/P"<>ToString@j<>"a.csv",Transpose[Xa]];
	Export["./Data/Synthetic/100/MultiStability/P"<>ToString@j<>"b.csv",Transpose[Xb]];
	Print[j],
{j,0,9,1}];]
