%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Place any paremeters here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TAU = 7;
EMBED = 7; 
NN = 10; 
THEILER = TAU*EMBED; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compile
mex arnhold.c

disp('Operation Started...')
X = (1:400)' + 0.1*randn(400, 1);
Y = 0.1*X + randn(400, 1);
XY = [X, Y];
arnhold(XY, TAU, EMBED, NN, THEILER)


% In your work use larger values for TAU and EMBED (e.g., TAU = 10, EMBED = 10).
