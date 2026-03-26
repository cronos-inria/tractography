#ifndef __UTILS_SPHARM__
#define __UTILS_SPHARM__

#include "utils/core.cl"

#define SOFTMAX_SCALE 100.0f

// Returns the value of the FOD in the specified direction.
//
float shval(__global const float fod[45], float4 direction) {

	float2 angles = cart2sph(direction);
	float phi = angles.x;
	float theta = angles.y;

	float st, ct;
	st = sincos(theta, &ct);
	
	float sp, cp, s2p, c2p, s3p, c3p, s4p, c4p, s5p, c5p, s6p, c6p, s7p, c7p, s8p, c8p;
	sp = sincos(phi, &cp);
	s2p = sincos(2.0f * phi, &c2p);
	
	// Use recursion instead of:
	// s3p = sincos(3.0f * phi, &c3p);
	s3p = 2 * cp * s2p - sp;
	c3p = 2 * cp * c2p - cp;
	// s4p = sincos(4.0f * phi, &c4p);
	s4p = 2 * cp * s3p - s2p;
	c4p = 2 * cp * c3p - c2p;
	// s5p = sincos(5.0f * phi, &c5p);
	s5p = 2 * cp * s4p - s3p;
	c5p = 2 * cp * c4p - c3p;
	// s6p = sincos(6.0f * phi, &c6p);
	s6p = 2 * cp * s5p - s4p;
	c6p = 2 * cp * c5p - c4p;
	// s7p = sincos(7.0f * phi, &c7p);
	s7p = 2 * cp * s6p - s5p;
	c7p = 2 * cp * c6p - c5p;
	// s8p = sincos(8.0f * phi, &c8p);
	s8p = 2 * cp * s7p - s6p;
	c8p = 2 * cp * c7p - c6p;

	// Precompute some values.
	float st2 = st * st;
	float st3 = st2 * st;
	float st4 = st3 * st;
	float st5 = st4 * st;
	float st6 = st5 * st;
	float st7 = st6 * st;
	float st8 = st7 * st;
	float ct2 = ct * ct;
	float ct3 = ct2 * ct;
	float ct4 = ct3 * ct;
	float ct5 = ct4 * ct;
	float ct6 = ct5 * ct;
	float ct7 = ct6 * ct;
	float ct8 = ct7 * ct;
	float stct = st * ct;

    float value = fod[0] * 0.28209479177387814f;

    value += fod[1] * 0.54627421529f * st2 * s2p;
    value += fod[2] * -1.0925484305920792f * stct * sp;
    value += fod[3] * 0.31539156525252005f * (3.0f * ct2 - 1.0f);
    value += fod[4] * -1.0925484305920792f * stct * cp;
    value += fod[5] *  0.54627421529f * st2 * c2p;

    value += fod[6] * 0.62583573544f * st4 * s4p;
    value += fod[7] * -1.77013076978f * st3 * ct * s3p;
    value += fod[8] * 0.47308734787f * st2 * (7.0f * ct2 - 1.0f) * s2p;
    value += fod[9] * -0.66904654355f * st * (7.0f * ct2 - 3.0f) * ct * sp;
    value += fod[10] *  0.10578554691520431f * (35.0f * ct2 * ct2 - 30.0f * ct2 + 3.0f);
    value += fod[11] *  -0.66904654355f * st * (7.0f * ct2 - 3.0f) * ct * cp;
    value += fod[12] *  0.47308734787f * st2 * (7.0f * ct2 - 1.0f) * c2p;
    value += fod[13] *  -1.77013076978f * st3 * ct * c3p;
    value += fod[14] *  0.62583573544f * st4 * c4p;

	value += fod[15] *  0.68318410519f * st6 * s6p;
	value += fod[16] *  -2.36661916223f * st5 * ct * s5p;
	value += fod[17] *  0.50456490072f * st4 * (11.0f * ct2 - 1.0f) * s4p;
	value += fod[18] *  -0.92120525951f * st3 * (11.0f * ct3 - 3.0f * ct) * s3p;
	value += fod[19] *  0.46060262975f * st2 * (33.0f * ct4 - 18.0f * ct2 + 1.0f) * s2p;
	value += fod[20] *  -0.58262136251f * st * (33.0f * ct5 - 30.0f * ct3 + 5.0f * ct) * sp;
	value += fod[21] *  0.06356920226f * (231.0f * ct6 - 315.0f * ct4 + 105.0f * ct2 - 5.0f);
	value += fod[22] *  -0.58262136251f * st * (33.0f * ct5 - 30.0f * ct3 + 5.0f * ct) * cp;
	value += fod[23] *  0.46060262975f * st2 * (33.0f * ct4 - 18.0f * ct2 + 1.0f) * c2p;
	value += fod[24] *  -0.92120525951f * st3 * (11.0f * ct3 - 3.0f * ct) * c3p;
	value += fod[25] *  0.50456490072f * st4 * (11.0f * ct2 - 1.0f) * c4p;
	value += fod[26] *  -2.36661916223f * st5 * ct * c5p;
	value += fod[27] *  0.68318410519f * st6 * c6p;

	value += fod[28] *  0.72892666017f * st8 * s8p;
	value += fod[29] *  -2.9157066407f * st7 * ct * s7p;
	value += fod[30] *  0.53233276606f * st6 * (15.0f * ct2 - 1.0f) * s6p;
	value += fod[31] *  -3.4499106221f * st5 * (5.0f * ct3 - ct) * s5p;
	value += fod[32] *  0.47841652475f * st4 * (65.0f * ct4 - 26.0f * ct2 + 1.0f) * s4p;
	value += fod[33] *  -1.2352661553f * st3 * (39.0f * ct5 - 26.0f * ct3 + 3 * ct) * s3p;
	value += fod[34] *  0.45615225843f * st2 * (143.0f * ct6 - 143.0f * ct4 + 33.0f * ct2 - 1.0f) * s2p;
	value += fod[35] *  -0.10904124589f * st * (715.0f * ct7 - 1001.0f * ct5 + 385.0f * ct3 - 35.0f * ct) * sp;
	value += fod[36] *  0.00908677049f * (6435.0f * ct8 - 12012.0f * ct6 + 6930.0f * ct4 - 1260.0f * ct2 + 35.0f); 
	value += fod[37] *  -0.10904124589f * st * (715.0f * ct7 - 1001.0f * ct5 + 385.0f * ct3 - 35.0f * ct) * cp;
	value += fod[38] *  0.45615225843f * st2 * (143.0f * ct6 - 143.0f * ct4 + 33.0f * ct2 - 1.0f) * c2p;
	value += fod[39] *  -1.2352661553f * st3 * (39.0f * ct5 - 26.0f * ct3 + 3 * ct) * c3p;
	value += fod[40] *  0.47841652475f * st4 * (65.0f * ct4 - 26.0f * ct2 + 1.0f) * c4p;
	value += fod[41] *  -3.4499106221f * st5 * (5.0f * ct3 - ct) * c5p;
	value += fod[42] *  0.53233276606f * st6 * (15.0f * ct2 - 1.0f) * c6p;
	value += fod[43] *  -2.9157066407f * st7 * ct * c7p; 
	value += fod[44] *  0.72892666017f * st8 * c8p; 

	return value;
}


void ishtmtx(float phi, float theta, float ylm[45], float ylm_dp[45], float ylm_dt[45]) {

	float st, ct;
	st = sincos(theta, &ct);
	
	float sp, cp, s2p, c2p, s3p, c3p, s4p, c4p, s5p, c5p, s6p, c6p, s7p, c7p, s8p, c8p;
	sp = sincos(phi, &cp);
	s2p = sincos(2.0f * phi, &c2p);
	
	// Use recursion instead of:
	// s3p = sincos(3.0f * phi, &c3p);
	s3p = 2 * cp * s2p - sp;
	c3p = 2 * cp * c2p - cp;
	// s4p = sincos(4.0f * phi, &c4p);
	s4p = 2 * cp * s3p - s2p;
	c4p = 2 * cp * c3p - c2p;
	// s5p = sincos(5.0f * phi, &c5p);
	s5p = 2 * cp * s4p - s3p;
	c5p = 2 * cp * c4p - c3p;
	// s6p = sincos(6.0f * phi, &c6p);
	s6p = 2 * cp * s5p - s4p;
	c6p = 2 * cp * c5p - c4p;
	// s7p = sincos(7.0f * phi, &c7p);
	s7p = 2 * cp * s6p - s5p;
	c7p = 2 * cp * c6p - c5p;
	// s8p = sincos(8.0f * phi, &c8p);
	s8p = 2 * cp * s7p - s6p;
	c8p = 2 * cp * c7p - c6p;

	// Precompute some values.
	float st2 = st * st;
	float st3 = st2 * st;
	float st4 = st3 * st;
	float st5 = st4 * st;
	float st6 = st5 * st;
	float st7 = st6 * st;
	float st8 = st7 * st;
	float ct2 = ct * ct;
	float ct3 = ct2 * ct;
	float ct4 = ct3 * ct;
	float ct5 = ct4 * ct;
	float ct6 = ct5 * ct;
	float ct7 = ct6 * ct;
	float ct8 = ct7 * ct;
	float stct = st * ct;

    ylm[0] = 0.28209479177387814f;

    ylm[1] = 0.54627421529f * st2 * s2p;
    ylm[2] = -1.0925484305920792f * stct * sp;
    ylm[3] = 0.31539156525252005f * (3.0f * ct2 - 1.0f);
    ylm[4] = -1.0925484305920792f * stct * cp;
    ylm[5] =  0.54627421529f * st2 * c2p;

    ylm[6] = 0.62583573544f * st4 * s4p;
    ylm[7] = -1.77013076978f * st3 * ct * s3p;
    ylm[8] = 0.47308734787f * st2 * (7.0f * ct2 - 1.0f) * s2p;
    ylm[9] = -0.66904654355f * st * (7.0f * ct2 - 3.0f) * ct * sp;
    ylm[10] = 0.10578554691520431f * (35.0f * ct2 * ct2 - 30.0f * ct2 + 3.0f);
    ylm[11] = -0.66904654355f * st * (7.0f * ct2 - 3.0f) * ct * cp;
    ylm[12] = 0.47308734787f * st2 * (7.0f * ct2 - 1.0f) * c2p;
    ylm[13] = -1.77013076978f * st3 * ct * c3p;
    ylm[14] = 0.62583573544f * st4 * c4p;

	ylm[15] = 0.68318410519f * st6 * s6p;
	ylm[16] = -2.36661916223f * st5 * ct * s5p;
	ylm[17] = 0.50456490072f * st4 * (11.0f * ct2 - 1.0f) * s4p;
	ylm[18] = -0.92120525951f * st3 * (11.0f * ct3 - 3.0f * ct) * s3p;
	ylm[19] = 0.46060262975f * st2 * (33.0f * ct4 - 18.0f * ct2 + 1.0f) * s2p;
	ylm[20] = -0.58262136251f * st * (33.0f * ct5 - 30.0f * ct3 + 5.0f * ct) * sp;
	ylm[21] = 0.06356920226f * (231.0f * ct6 - 315.0f * ct4 + 105.0f * ct2 - 5.0f);
	ylm[22] = -0.58262136251f * st * (33.0f * ct5 - 30.0f * ct3 + 5.0f * ct) * cp;
	ylm[23] = 0.46060262975f * st2 * (33.0f * ct4 - 18.0f * ct2 + 1.0f) * c2p;
	ylm[24] = -0.92120525951f * st3 * (11.0f * ct3 - 3.0f * ct) * c3p;
	ylm[25] = 0.50456490072f * st4 * (11.0f * ct2 - 1.0f) * c4p;
	ylm[26] = -2.36661916223f * st5 * ct * c5p;
	ylm[27] = 0.68318410519f * st6 * c6p;

	ylm[28] = 0.72892666017f * st8 * s8p;
	ylm[29] = -2.9157066407f * st7 * ct * s7p;
	ylm[30] = 0.53233276606f * st6 * (15.0f * ct2 - 1.0f) * s6p;
	ylm[31] = -3.4499106221f * st5 * (5.0f * ct3 - ct) * s5p;
	ylm[32] = 0.47841652475f * st4 * (65.0f * ct4 - 26.0f * ct2 + 1.0f) * s4p;
	ylm[33] = -1.2352661553f * st3 * (39.0f * ct5 - 26.0f * ct3 + 3 * ct) * s3p;
	ylm[34] = 0.45615225843f * st2 * (143.0f * ct6 - 143.0f * ct4 + 33.0f * ct2 - 1.0f) * s2p;
	ylm[35] = -0.10904124589f * st * (715.0f * ct7 - 1001.0f * ct5 + 385.0f * ct3 - 35.0f * ct) * sp;
	ylm[36] = 0.00908677049f * (6435.0f * ct8 - 12012.0f * ct6 + 6930.0f * ct4 - 1260.0f * ct2 + 35.0f); 
	ylm[37] = -0.10904124589f * st * (715.0f * ct7 - 1001.0f * ct5 + 385.0f * ct3 - 35.0f * ct) * cp;
	ylm[38] = 0.45615225843f * st2 * (143.0f * ct6 - 143.0f * ct4 + 33.0f * ct2 - 1.0f) * c2p;
	ylm[39] = -1.2352661553f * st3 * (39.0f * ct5 - 26.0f * ct3 + 3 * ct) * c3p;
	ylm[40] = 0.47841652475f * st4 * (65.0f * ct4 - 26.0f * ct2 + 1.0f) * c4p;
	ylm[41] = -3.4499106221f * st5 * (5.0f * ct3 - ct) * c5p;
	ylm[42] = 0.53233276606f * st6 * (15.0f * ct2 - 1.0f) * c6p;
	ylm[43] = -2.9157066407f * st7 * ct * c7p; 
	ylm[44] = 0.72892666017f * st8 * c8p; 

	// Ylm derivative with respect to the azimuth (phi).
    ylm_dp[0] = 0.0f;

    ylm_dp[1] = 0.54627421529f * st2 * 2.0f * c2p;
    ylm_dp[2] = -1.0925484305920792f * stct * cp;
    ylm_dp[3] = 0.0f; 
    ylm_dp[4] = 1.0925484305920792f * stct * sp;
    ylm_dp[5] =  -0.54627421529f * st2 * 2.0f * s2p;

    ylm_dp[6] = 0.62583573544f * st4 * 4.0f * c4p;
    ylm_dp[7] = -1.77013076978f * st3 * ct * 3.0f * c3p;
    ylm_dp[8] = 0.47308734787f * st2 * (7.0f * ct2 - 1.0f) * 2.0f * c2p;
    ylm_dp[9] = -0.66904654355f * stct * (7.0f * ct2 - 3.0f) * cp;
    ylm_dp[10] = 0.0f;
    ylm_dp[11] = 0.66904654355f * stct * (7.0f * ct2 - 3.0f) * sp;
    ylm_dp[12] = -0.47308734787f * st2 * (7.0f * ct2 - 1.0f) * 2.0f * s2p;
    ylm_dp[13] = 1.77013076978f * st3 * ct * 3.0f * s3p;
    ylm_dp[14] = -0.62583573544f * st4 * 4.0f * s4p;

	ylm_dp[15] = 0.68318410519f * st6 * 6.0f * c6p;
	ylm_dp[16] = -2.36661916223f * st5 * ct * 5.0f * c5p;
	ylm_dp[17] = 0.50456490072f * st4 * (11.0f * ct2 - 1.0f) * 4.0f * c4p;
	ylm_dp[18] = -0.92120525951f * st3 * (11.0f * ct3 - 3.0f * ct) * 3.0f * c3p;
	ylm_dp[19] = 0.46060262975f * st2 * (33.0f * ct4 - 18.0f * ct2 + 1.0f) * 2.0f * c2p;
	ylm_dp[20] = -0.58262136251f * st * (33.0f * ct5 - 30.0f * ct3 + 5.0f * ct) * cp;
	ylm_dp[21] = 0.0f; 
	ylm_dp[22] = 0.58262136251f * st * (33.0f * ct5 - 30.0f * ct3 + 5.0f * ct) * sp;
	ylm_dp[23] = -0.46060262975f * st2 * (33.0f * ct4 - 18.0f * ct2 + 1.0f) * 2.0f * s2p;
	ylm_dp[24] = 0.92120525951f * st3 * (11.0f * ct3 - 3.0f * ct) * 3.0f * s3p;
	ylm_dp[25] = -0.50456490072f * st4 * (11.0f * ct2 - 1.0f) * 4.0f * s4p;
	ylm_dp[26] = 2.36661916223f * st5 * ct * 5.0f * s5p;
	ylm_dp[27] = -0.68318410519f * st6 * 6.0f * s6p;

	ylm_dp[28] = 0.72892666017f * st8 * 8.0f * c8p;
	ylm_dp[29] = -2.9157066407f * st7 * ct * 7.0f * c7p;
	ylm_dp[30] = 0.53233276606f * st6 * (15.0f * ct2 - 1.0f) * 6.0f * c6p;
	ylm_dp[31] = -3.4499106221f * st5 * (5.0f * ct3 - ct) * 5.0f * c5p;
	ylm_dp[32] = 0.47841652475f * st4 * (65.0f * ct4 - 26.0f * ct2 + 1.0f) * 4.0f * c4p;
	ylm_dp[33] = -1.2352661553f * st3 * (39.0f * ct5 - 26.0f * ct3 + 3 * ct) * 3.0f * c3p;
	ylm_dp[34] = 0.45615225843f * st2 * (143.0f * ct6 - 143.0f * ct4 + 33.0f * ct2 - 1.0f) * 2.0f * c2p;
	ylm_dp[35] = -0.10904124589f * st * (715.0f * ct7 - 1001.0f * ct5 + 385.0f * ct3 - 35.0f * ct) * cp;
	ylm_dp[36] = 0.0f; 
	ylm_dp[37] = 0.10904124589f * st * (715.0f * ct7 - 1001.0f * ct5 + 385.0f * ct3 - 35.0f * ct) * sp;
	ylm_dp[38] = -0.45615225843f * st2 * (143.0f * ct6 - 143.0f * ct4 + 33.0f * ct2 - 1.0f) * 2.0f * s2p;
	ylm_dp[39] = 1.2352661553f * st3 * (39.0f * ct5 - 26.0f * ct3 + 3 * ct) * 3.0f * s3p;
	ylm_dp[40] = -0.47841652475f * st4 * (65.0f * ct4 - 26.0f * ct2 + 1.0f) * 4.0f * s4p;
	ylm_dp[41] = 3.4499106221f * st5 * (5.0f * ct3 - ct) * 5.0f * s5p;
	ylm_dp[42] = -0.53233276606f * st6 * (15.0f * ct2 - 1.0f) * 6.0f * s6p;
	ylm_dp[43] = 2.9157066407f * st7 * ct * 7.0f * s7p; 
	ylm_dp[44] = -0.72892666017f * st8 * 8.0f * s8p; 

	// Ylm derivative with respect to the colatitude (theta).
    ylm_dt[0] = 0.0f;

    ylm_dt[1] = 0.54627421529f * 2.0f * stct * s2p;
    ylm_dt[2] = -1.0925484305920792f * (ct2 - st2) * sp;
    ylm_dt[3] = -0.31539156525252005f * 6.0f * ct * st;
    ylm_dt[4] = -1.0925484305920792f * (ct2 - st2) * cp;
    ylm_dt[5] =  0.54627421529f * 2.0f * stct * c2p;

    ylm_dt[6] = 0.62583573544f * 4.0f * st3 * ct * s4p;
    ylm_dt[7] = -1.77013076978f * (3.0f * st2 * ct2 - st4) * s3p;
    ylm_dt[8] = 0.47308734787f * (2.0f * stct * (7.0f * ct2 - 1.0f) - 14.0f * st3 * ct) * s2p; 
    ylm_dt[9] = -0.66904654355f * ((ct2 - st2) * (7.0f * ct2 - 3.0f) - (14.0f * st2 * ct2)) * sp;
    ylm_dt[10] = 0.10578554691520431f * (-35.0f * 4.0f * ct2 * stct + 30.0f * 2.0f * stct);
    ylm_dt[11] = -0.66904654355f * ((ct2 - st2) * (7.0f * ct2 - 3.0f) - (14.0f * st2 * ct2)) * cp;
    ylm_dt[12] = 0.47308734787f * (2.0f * stct * (7.0f * ct2 - 1.0f) - 14.0f * st3 * ct) * c2p;
    ylm_dt[13] = -1.77013076978f * (3.0f * st2 * ct2 - st4) * c3p;
    ylm_dt[14] = 0.62583573544f * 4.0f * st3 * ct * c4p;

	ylm_dt[15] = 0.68318410519f * 6.0f * st5 * ct * s6p;
	ylm_dt[16] = -2.36661916223f * (5.0f * st4 * ct2 - st6) * s5p;
	ylm_dt[17] = 0.50456490072f * (4.0f * st3 * ct * (11.0f * ct2 - 1.0f) - 22.0f * st5 * ct) * s4p;
	ylm_dt[18] = -0.92120525951f * (3.0f * st2 * ct * (11.0f * ct3 - 3.0f * ct) - st4 * (33.0f * ct2 - 3.0f))* s3p;
	ylm_dt[19] = 0.46060262975f * (2.0f * st * ct * (33.0f * ct4 - 18.0f * ct2 + 1.0f) - st3 * (33.0f * 4.0f * ct3 - 36.0f * ct)) * s2p;
	ylm_dt[20] = -0.58262136251f * (ct * (33.0f * ct5 - 30.0f * ct3 + 5.0f * ct) - st2 * (33.0f * 5.0f * ct4 - 90.0f * ct2 + 5.0f)) * sp;
	ylm_dt[21] = -0.06356920226f * (1386.0f * ct5 - 1260.0f * ct3 + 210.0f * ct) * st;
	ylm_dt[22] = -0.58262136251f * (ct * (33.0f * ct5 - 30.0f * ct3 + 5.0f * ct) - st2 * (33.0f * 5.0f * ct4 - 90.0f * ct2 + 5.0f)) * cp;
	ylm_dt[23] = 0.46060262975f * (2.0f * st * ct * (33.0f * ct4 - 18.0f * ct2 + 1.0f) - st3 * (33.0f * 4.0f * ct3 - 36.0f * ct)) * c2p;
	ylm_dt[24] = -0.92120525951f * (3.0f * st2 * ct * (11.0f * ct3 - 3.0f * ct) - st4 * (33.0f * ct2 - 3.0f))* c3p;
	ylm_dt[25] = 0.50456490072f * (4.0f * st3 * ct * (11.0f * ct2 - 1.0f) - 22.0f * st5 * ct) * c4p;
	ylm_dt[26] = -2.36661916223f * (5.0f * st4 * ct2 - st6) * c5p;
	ylm_dt[27] = 0.68318410519f * 6.0f * st5 * ct * c6p;

	ylm_dt[28] = 0.72892666017f * 8.0f * st7 * ct * s8p;
	ylm_dt[29] = -2.9157066407f * (7.0f * st6 * ct2 - st8) * s7p;
	ylm_dt[30] = 0.53233276606f * (6.0f * st5 * ct * (15.0f * ct2 - 1.0f) - st7 * 30.0f * ct) * s6p;
	ylm_dt[31] = -3.4499106221f * (5.0f * st4 * ct * (5.0f * ct3 - ct) - st6 * (15.0f * ct2 - 1.0f)) * s5p;
	ylm_dt[32] = 0.47841652475f * (4.0f * st3 * ct * (65.0f * ct4 - 26.0f * ct2 + 1.0f) - st5 * (260.0f * ct3 - 52.0f * ct)) * s4p;
	ylm_dt[33] = -1.2352661553f * (3.0f * st2 * ct * (39.0f * ct5 - 26.0f * ct3 + 3 * ct) - st4 * (195.0f * ct4 - 78.0f * ct2 + 3.0f)) * s3p;
	ylm_dt[34] = 0.45615225843f * (2.0f * st * ct * (143.0f * ct6 - 143.0f * ct4 + 33.0f * ct2 - 1.0f) - st3 * (858.0f * ct5 - 572.0f * ct3 + 66.0f * ct)) * s2p;
	ylm_dt[35] = -0.10904124589f * (ct * (715.0f * ct7 - 1001.0f * ct5 + 385.0f * ct3 - 35.0f * ct) - st2 * (5005.0f * ct6 - 5005.0f * ct4 + 1155.0f * ct2 - 35.0f)) * sp;
	ylm_dt[36] = -0.00908677049f * (8.0f * 6435.0f * ct7 - 6.0f * 12012.0f * ct5 + 4.0f * 6930.0f * ct3 - 2.0f * 1260.0f * ct) * st; 
	ylm_dt[37] = -0.10904124589f * (ct * (715.0f * ct7 - 1001.0f * ct5 + 385.0f * ct3 - 35.0f * ct) - st2 * (5005.0f * ct6 - 5005.0f * ct4 + 1155.0f * ct2 - 35.0f)) * cp;
	ylm_dt[38] = 0.45615225843f * (2.0f * st * ct * (143.0f * ct6 - 143.0f * ct4 + 33.0f * ct2 - 1.0f) - st3 * (858.0f * ct5 - 572.0f * ct3 + 66.0f * ct)) * c2p;
	ylm_dt[39] = -1.2352661553f * (3.0f * st2 * ct * (39.0f * ct5 - 26.0f * ct3 + 3 * ct) - st4 * (195.0f * ct4 - 78.0f * ct2 + 3.0f)) * c3p;
	ylm_dt[40] = 0.47841652475f * (4.0f * st3 * ct * (65.0f * ct4 - 26.0f * ct2 + 1.0f) - st5 * (260.0f * ct3 - 52.0f * ct)) * c4p;
	ylm_dt[41] = -3.4499106221f * (5.0f * st4 * ct * (5.0f * ct3 - ct) - st6 * (15.0f * ct2 - 1.0f)) * c5p;
	ylm_dt[42] = 0.53233276606f * (6.0f * st5 * ct * (15.0f * ct2 - 1.0f) - st7 * 30.0f * ct) * c6p;
	ylm_dt[43] = -2.9157066407f * (7.0f * st6 * ct2 - st8) * c7p;
	ylm_dt[44] = 0.72892666017f * 8.0f * st7 * ct * c8p;
}

__kernel void test_ishtmtx(
	global const float* azimuth,
	global const float* colatitude,
	uint n_values,
	global float values[100][45])
{
	float ylm[45];
	float ylm_dp[45];
	float ylm_dt[45];
	for (size_t i = 0; i < n_values; i++) {
		ishtmtx(azimuth[i], colatitude[i], ylm, ylm_dp, ylm_dt);
		for (size_t j = 0; j < 45; j++) {
			values[i][j] = ylm[j];
		}
	}
}

__kernel void test_ishtmtx_dp(
	global const float* azimuth,
	global const float* colatitude,
	uint n_values,
	global float values[100][45])
{
	float ylm[45];
	float ylm_dp[45];
	float ylm_dt[45];
	for (size_t i = 0; i < n_values; i++) {
		ishtmtx(azimuth[i], colatitude[i], ylm, ylm_dp, ylm_dt);
		for (size_t j = 0; j < 45; j++) {
			values[i][j] = ylm_dp[j];
		}
	}
}

__kernel void test_ishtmtx_dt(
	global const float* azimuth,
	global const float* colatitude,
	uint n_values,
	global float values[100][45])
{
	float ylm[45];
	float ylm_dp[45];
	float ylm_dt[45];
	for (size_t i = 0; i < n_values; i++) {
		ishtmtx(azimuth[i], colatitude[i], ylm, ylm_dp, ylm_dt);
		for (size_t j = 0; j < 45; j++) {
			values[i][j] = ylm_dt[j];
		}
	}
}

#endif
