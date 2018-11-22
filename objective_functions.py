#!/usr/bin/env python3

from config import n, k, M
import numpy as np
import torch


def DTLZ1(x):
	F = torch.zeros((M, 0), dtype=torch.float32)
	gs = torch.sum(torch.pow(x[:, k-1:] - 0.5, 2.) - torch.cos(20. * np.pi * (x[:, k-1:] - 0.5)), dim=1)
	gs = 100. * ((n - k + 1.) + gs)

	for i in range(k):
		vals = 0.5 * (1. + gs)
		t = k - i - 1
		if t > 0:
			vals.mul_(torch.prod(x[:, :t], dim=1))
		if t < k - 1:
			vals.mul_(1. - x[:, t])
		F = torch.cat((F, vals.unsqueeze(1)), dim=1)
	return F


def DTLZ2(x):
	F = torch.zeros((M, 0), dtype=torch.float32)
	gs = torch.sum(torch.pow(x[:, k-1:] - 0.5, 2), dim=1)

	for i in range(k):
		vals = torch.ones(1) + gs
		t = k - i - 1
		if t > 0:
			vals.mul_(torch.prod(torch.cos(x[:, :t] * np.pi * 0.5), dim=1))
		if t < k - 1:
			vals.mul_(torch.sin(x[:, t] * np.pi * 0.5))
		F = torch.cat((F, vals.unsqueeze(1)), dim=1)
	return F


def DTLZ3(x):
	F = torch.zeros((M, 0), dtype=torch.float32)
	gs = torch.sum(torch.pow(x[:, k-1:] - 0.5, 2) - torch.cos(20. * np.pi * (x[:, k-1:] - 0.5)), dim=1)
	gs = 100. * (n - k + 1 + gs)

	for i in range(k):
		vals = 1. + gs
		t = k - i - 1
		if t > 0:
			vals.mul_(torch.prod(torch.cos(x[:, :t] * np.pi * 0.5), dim=1))
		if t < k - 1:
			vals.mul_(torch.sin(x[:, t] * np.pi * 0.5))
		F = torch.cat((F, vals.unsqueeze(1)), dim=1)
	return F


def DTLZ4(x):
	alpha = 100.
	F = torch.zeros((M, 0), dtype=torch.float32)
	gs = torch.sum(torch.pow(x[:, k-1:] - 0.5, 2), dim=1)

	for i in range(k):
		vals = torch.ones(1) + gs
		t = k - i - 1
		if t > 0:
			vals.mul_(torch.prod(torch.cos(torch.pow(x[:, :t], alpha) * np.pi * 0.5), dim=1))
		if t < k - 1:
			vals.mul_(torch.sin(torch.pow(x[:, t], alpha) * np.pi * 0.5))
		F = torch.cat((F, vals.unsqueeze(1)), dim=1)
	return F


def DTLZ5(x):
	F = torch.zeros((M, 0), dtype=torch.float32)
	gs = torch.sum(torch.pow(x[:, k-1:] - 0.5, 2), dim=1)
	p = np.pi / (4. * (1. + gs))

	thetas = torch.cat(
		((x[:, 0] * np.pi * 0.5).unsqueeze(1),
		p.unsqueeze(1) * (1 + 2 * gs.unsqueeze(1) * x[:, 1:k-1])),
		dim=1)

	for i in range(k):
		vals = torch.ones(1) + gs
		t = k - i - 1
		if t > 0:
			vals.mul_(torch.prod(torch.cos(thetas[:, :t]), dim=1))
		if t < k - 1:
			vals.mul_(torch.sin(thetas[:, t]))
		F = torch.cat((F, vals.unsqueeze(1)), dim=1)
	return F


def DTLZ6(x):
	F = torch.zeros((M, 0), dtype=torch.float32)
	gs = torch.sum(torch.pow(x[:, k-1:], 0.1), dim=1)
	p = np.pi / (4. * (1. + gs))

	thetas = (x[:, 0] * np.pi * 0.5).unsqueeze(1)
	thetas = torch.cat((thetas, p.unsqueeze(1) * (1 + 2 * gs.unsqueeze(1) * x[:, 1:k-1])), dim=1)

	for i in range(k):
		vals = torch.ones(1) + gs
		t = k - i - 1
		if t > 0:
			vals.mul_(torch.prod(torch.cos(thetas[:, :t]), dim=1))
		if t < k - 1:
			vals.mul_(torch.sin(thetas[:, t]))
		F = torch.cat((F, vals.unsqueeze(1)), dim=1)
	return F


def DTLZ7(x):
	F = x[:, :k-1]
	gs = 1. + 9. * torch.sum(x[:, k-1:], dim=1) / (n - k + 1)
	hs = k - torch.sum(x[:, :k-1] / (1. + gs).unsqueeze(1) * (1. + torch.sin(3. * np.pi * x[:, :k-1])), dim=1)
	F = torch.cat((F, ((1. + gs) * hs).unsqueeze(1)), dim=1)
	return F