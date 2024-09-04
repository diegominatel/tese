######################################################################################
### Exemplo utilizado na Fundamentação de Teoria da Resposta ao Item (Seção 2.3.4) ###
######################################################################################

# Carrega o pacote mirt, responsável pelos métdos da TRI no R
library(mirt)

# Respostas dos itens dos 8 examinados
Item1 <- c(0, 1, 0, 0, 1, 1, 0, 1)
Item2 <- c(0, 0, 1, 0, 1, 0, 1, 1)
Item3 <- c(0, 0, 0, 1, 0, 1, 1, 1)
# Transforma em matriz, de acordo com o mirt
data <- matrix(c(Item1, Item2, Item3), ncol=3)
# Nomeia as colunas
colnames(data) <- c('Item 1', 'Item 2', 'Item 3')
# Visualiza os dados
dataset

#####
# Calcula os valores de Theta para o ML1
# Os parâmetros são definidos conforme enunciada na Seção da TRI na Tese
#####

# Define que o modelo vai ter valores predefinidos
params <- mirt(dataset, 1, 'Rasch', pars='values')
# Desabilita a estimação do parâmetro b para todos os itens
params$est[2] = FALSE
params$est[6] = FALSE
params$est[10] = FALSE
# Define os valores para cada item do parâmetro b
# Lembrando que para definir o valor é estima d = (-1)*b*a 
# Portanto tem que inserir d, pois é o que é estimado de fato
# Como o ML1 a = 1, temos que d = (-1)*b
params$value[2] = (-1)*-2
params$value[6] = (-1)*0
params$value[10] = (-1)*2
# Estima os valores de theta
ml1 = mirt(data, 1, pars = params)
coefs = coef(ml1, simplify=TRUE, IRTpars=TRUE)
# Visualiza se os parâmetros estão corretos
coefs$items
# Mostra os valores de theta
fscores(myObj)

#####
# Calcula os valores de Theta para o ML2
# Os parâmetros são definidos conforme enunciada na Seção da TRI na Tese
#####

# Define que o modelo vai ter valores predefinido
params <- mirt(dataset, 1, '2PL', pars='values')
# Desabilita a estimação dos parâmetros a e b para todos os itens
params$est[1] = FALSE
params$est[2] = FALSE
params$est[5] = FALSE
params$est[6] = FALSE
params$est[9] = FALSE
params$est[10] = FALSE
# Define os valores para cada item dos parâmetros a e b
# Lembrando que para definir o valor é estima d = (-1)*b*a 
# Portanto tem que inserir d, pois é o que é estimado de fato
params$value[1] = 2
params$value[2] = (-1)*-2*2
params$value[5] = 1
params$value[6] = ((-1)*0)*1
params$value[9] = 0.5
params$value[10] = ((-1)*2)*0.5
# Estima os valores de theta
ml2 = mirt(data, 1, pars = params)
coefs = coef(ml2, simplify=TRUE, IRTpars=TRUE)
# Visualiza se os parâmetros estão corretos
coefs$items
# Mostra os valores de theta
fscores(ml2)

#####
# Calcula os valores de Theta para o ML3
# Os parâmetros são definidos conforme enunciada na Seção da TRI na Tese
#####

# Define que o modelo vai ter valores predefinido
params <- mirt(dataset, 1, '3PL', pars='values')
# Desabilita a estimação dos parâmetros a, b, c para todos os itens
params$est[1] = FALSE
params$est[2] = FALSE
params$est[3] = FALSE
params$est[5] = FALSE
params$est[6] = FALSE
params$est[7] = FALSE
params$est[9] = FALSE
params$est[10] = FALSE
params$est[11] = FALSE
# Define os valores para cada item dos parâmetros a, b e c
# Lembrando que para definir o valor é estima d = (-1)*b*a 
# Portanto tem que inserir d, pois é o que é estimado de fato
params$value[1] = 2
params$value[2] = (-1)*-2*2
params$value[3] = 0.5 
params$value[5] = 1
params$value[6] = ((-1)*0)*1
params$value[7] = 0.5
params$value[9] = 0.5
params$value[10] = ((-1)*2)*0.5
params$value[11] = 0.5
# Estima os valores de theta
ml3 = mirt(data, 1, pars = params)
coefs = coef(ml3, simplify=TRUE, IRTpars=TRUE)
# Visualiza se os parâmetros estão corretos
coefs$items
# Mostra os valores de theta
fscores(ml3)
