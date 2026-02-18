library(mgcv) # version >= 1.8-34
library(data.table)

set.seed(42)

n = 200; k = 30; bs_dim = 15; l_scale = 2
data = data.table(x = seq(-1, 1, l=n))
data[, y := x^2 + rnorm(n, sd=0.25)]

model = gam(y ~ s(x, bs="gp", k=bs_dim, m=c(-4, l_scale),
                     xt=list(max.knots=k)), data=data)

get_gp_params <- function(smooth, gp_coefs) {

    Qc = qr.Q(attr(smooth, "qrc"), complete=T) %*% c(0, gp_coefs)
    Kmm_inverse_u = smooth$UZ %*% Qc[-nrow(Qc),, drop=F]

    K_nm = sweep(matrix(data$x, n, 1), 2, smooth$shift)
    K_nm = (matrix(K_nm, n, k) - matrix(smooth$knt, n, k, T))^2
    K_nm = sqrt(K_nm)/smooth$gp.defn[2]
    K_nm = exp(-K_nm) + (K_nm * exp(-K_nm)) * (1 + K_nm/3)

    K_mm = (matrix(smooth$knt, k, k) - matrix(smooth$knt, k, k, T))^2
    K_mm = sqrt(K_mm)/smooth$gp.defn[2]
    K_mm = exp(-K_mm) + (K_mm * exp(-K_mm)) * (1 + K_mm/3)

    return(list(K_nm=K_nm, Kmm_inverse_u=Kmm_inverse_u,
                K_mm=K_mm, intercept=Qc[nrow(Qc)]))
}

gp = get_gp_params(model$smooth[[1]], coef(model)[-1])
mu = gp$K_nm %*% gp$Kmm_inverse_u + gp$intercept

max(abs(mu[, 1] + coef(model)["(Intercept)"] - predict(model))) # 1e-11