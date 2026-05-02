#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../engine/dense_layer.hpp"
#include "../engine/network.hpp"

namespace {

bool nearly_equal(float a, float b, float tol) {
	return std::fabs(a - b) <= tol;
}

void require(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "Test failure: %s\n", message);
		std::exit(1);
	}
}

float run_loss(Dense& layer, const std::vector<float>& input, size_t batch) {
	FloatBuffer input_buf(input.size());
	FloatBuffer output_buf(batch * static_cast<size_t>(layer.out_features()));
	input_buf.copy_from_host(input.data(), input.size());
	layer.forward(input_buf, output_buf);

	std::vector<float> output_host(output_buf.size());
	output_buf.copy_to_host(output_host.data(), output_host.size());

	float loss = 0.0f;
	for (float value : output_host) {
		loss += value;
	}
	return loss;
}

void test_dense_shapes_and_xavier() {
	const int in_features = 784;
	const int out_features = 128;
	const size_t batch = 32;

	Dense layer(in_features, out_features);
	FloatBuffer input(batch * static_cast<size_t>(in_features));
	FloatBuffer output(batch * static_cast<size_t>(out_features));
	FloatBuffer grad_out(batch * static_cast<size_t>(out_features));
	FloatBuffer grad_in(batch * static_cast<size_t>(in_features));

	std::vector<float> input_host(input.size(), 0.01f);
	std::vector<float> grad_out_host(grad_out.size(), 1.0f);
	input.copy_from_host(input_host.data(), input_host.size());
	grad_out.copy_from_host(grad_out_host.data(), grad_out_host.size());

	layer.forward(input, output);
	layer.backward(grad_out, grad_in);

	require(output.size() == batch * static_cast<size_t>(out_features),
			"Dense output shape mismatch");
	require(grad_in.size() == batch * static_cast<size_t>(in_features),
			"Dense grad_in shape mismatch");

	auto params = layer.parameters();
	std::vector<float> weights_host(params[0]->size());
	params[0]->copy_to_host(weights_host.data(), weights_host.size());

	float max_abs = 0.0f;
	bool has_zero = false;
	for (float value : weights_host) {
		max_abs = std::max(max_abs, std::fabs(value));
		if (value == 0.0f) {
			has_zero = true;
		}
	}

	std::printf("Xavier init check: max |W| = %.6f\n", max_abs);
	std::printf("Sample W values: %.6f %.6f %.6f %.6f %.6f\n",
				weights_host[0], weights_host[1], weights_host[2],
				weights_host[3], weights_host[4]);

	require(!has_zero, "Xavier init produced zeros");
	require(max_abs <= 0.2f, "Xavier init exceeded 0.2 bound");
}

void test_dense_hand_computed_forward() {
	Dense layer(4, 2);
	auto params = layer.parameters();

	std::vector<float> weights = {
		1.0f, 2.0f, 3.0f, 4.0f,
		-1.0f, 0.5f, 0.0f, 2.0f
	};
	std::vector<float> bias = {0.5f, -0.5f};
	params[0]->copy_from_host(weights.data(), weights.size());
	params[1]->copy_from_host(bias.data(), bias.size());

	std::vector<float> input_host = {1.0f, -1.0f, 2.0f, 0.5f};
	FloatBuffer input(input_host.size());
	FloatBuffer output(2);
	input.copy_from_host(input_host.data(), input_host.size());

	layer.forward(input, output);

	std::vector<float> out_host(output.size());
	output.copy_to_host(out_host.data(), out_host.size());

	require(nearly_equal(out_host[0], 7.5f, 1e-5f), "Dense forward value mismatch");
	require(nearly_equal(out_host[1], 0.0f, 1e-5f), "Dense forward ReLU mismatch");
}

void test_dense_numerical_gradient() {
	Dense layer(3, 2);
	auto params = layer.parameters();

	std::vector<float> weights = {
		0.05f, 0.04f, 0.03f,
		0.06f, 0.02f, 0.01f
	};
	std::vector<float> bias = {0.1f, 0.1f};
	params[0]->copy_from_host(weights.data(), weights.size());
	params[1]->copy_from_host(bias.data(), bias.size());

	const size_t batch = 2;
	std::vector<float> input = {
		0.5f, 0.4f, 0.3f,
		0.2f, 0.1f, 0.05f
	};
	FloatBuffer input_buf(input.size());
	FloatBuffer output_buf(batch * static_cast<size_t>(layer.out_features()));
	FloatBuffer grad_out_buf(output_buf.size());
	FloatBuffer grad_in_buf(batch * static_cast<size_t>(layer.in_features()));

	std::vector<float> grad_out_host(output_buf.size(), 1.0f);
	input_buf.copy_from_host(input.data(), input.size());
	grad_out_buf.copy_from_host(grad_out_host.data(), grad_out_host.size());

	layer.forward(input_buf, output_buf);
	layer.backward(grad_out_buf, grad_in_buf);

	auto grads = layer.gradients();
	std::vector<float> dweights_host(grads[0]->size());
	grads[0]->copy_to_host(dweights_host.data(), dweights_host.size());

	const float eps = 1e-3f;
	for (size_t i = 0; i < weights.size(); ++i) {
		std::vector<float> weights_plus = weights;
		std::vector<float> weights_minus = weights;
		weights_plus[i] += eps;
		weights_minus[i] -= eps;

		params[0]->copy_from_host(weights_plus.data(), weights_plus.size());
		float loss_plus = run_loss(layer, input, batch);

		params[0]->copy_from_host(weights_minus.data(), weights_minus.size());
		float loss_minus = run_loss(layer, input, batch);

		float numeric = (loss_plus - loss_minus) / (2.0f * eps);
		float analytic = dweights_host[i];
		require(nearly_equal(numeric, analytic, 1e-2f), "Dense gradient check failed");
	}

	params[0]->copy_from_host(weights.data(), weights.size());
}

} // namespace

int main() {
	std::printf("Running Dense layer tests...\n");
	test_dense_shapes_and_xavier();
	std::printf("Shape and Xavier test passed.\n");
	test_dense_hand_computed_forward();
	std::printf("Hand-computed forward test passed.\n");
	test_dense_numerical_gradient();
	std::printf("Numerical gradient test passed.\n");

	std::printf("Running Network training test...\n");
	const size_t batch = 100;
	const int in_features = 3072;
	const int hidden = 512;
	const int out = 10;

	std::mt19937 rng(2026);
	std::normal_distribution<float> dist(0.0f, 1.0f);
	std::uniform_int_distribution<int> label_dist(0, out - 1);

	std::vector<float> input(batch * static_cast<size_t>(in_features));
	std::vector<int> labels(batch);
	for (float& value : input) {
		value = dist(rng);
	}
	for (int& label : labels) {
		label = label_dist(rng);
	}

	FloatBuffer input_buf(input.size());
	LabelBuffer label_buf(labels.size());
	input_buf.copy_from_host(input.data(), input.size());
	label_buf.copy_from_host(labels.data(), labels.size());

	std::vector<Dense> layers;
	layers.emplace_back(in_features, hidden);
	layers.emplace_back(hidden, out);
	Network net(std::move(layers));

	float initial_loss = net.forward(input_buf, label_buf);
	std::printf("Step 0 loss: %.4f\n", initial_loss);
	require(initial_loss > 2.1f && initial_loss < 2.5f,
			"Initial loss not in expected range");

	float last_loss = initial_loss;
	for (int step = 1; step <= 50; ++step) {
		net.backward();
		net.sgd_step(0.01f, nullptr);
		last_loss = net.forward(input_buf, label_buf);
		if (step % 10 == 0) {
			std::printf("Step %d loss: %.4f\n", step, last_loss);
		}
	}
	require(last_loss < 1.0f, "Loss did not fall below 1.0");

	std::vector<float> eval_input(batch * static_cast<size_t>(in_features));
	std::vector<int> eval_labels(batch);
	for (float& value : eval_input) {
		value = dist(rng);
	}
	for (int& label : eval_labels) {
		label = label_dist(rng);
	}
	FloatBuffer eval_input_buf(eval_input.size());
	LabelBuffer eval_label_buf(eval_labels.size());
	eval_input_buf.copy_from_host(eval_input.data(), eval_input.size());
	eval_label_buf.copy_from_host(eval_labels.data(), eval_labels.size());

	float acc = net.get_accuracy(eval_input_buf, eval_label_buf);
	std::printf("Accuracy on random labels: %.2f%%\n", acc * 100.0f);
	require(acc > 0.05f && acc < 0.20f, "Accuracy not near random chance");

	std::printf("Network training test passed.\n");
	std::printf("All Dense layer tests passed.\n");
	return 0;
}
