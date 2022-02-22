#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include "benchmarking.h"
#include "assignment.h"

#define DECLARATION "In Congress, July 4, 1776\n\nThe unanimous Declaration of the thirteen united States of America, When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.\n\nWe hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, --That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that Governments long established should not be changed for light and transient causes; and accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils are sufferable, than to right themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a design to reduce them under absolute Despotism, it is their right, it is their duty, to throw off such Government, and to provide new Guards for their future security.--Such has been the patient sufferance of these Colonies; and such is now the necessity which constrains them to alter their former Systems of Government. The history of the present King of Great Britain is a history of repeated injuries and usurpations, all having in direct object the establishment of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid world.\n\nHe has refused his Assent to Laws, the most wholesome and necessary for the public good.\n\nHe has forbidden his Governors to pass Laws of immediate and pressing importance, unless suspended in their operation till his Assent should be obtained; and when so suspended, he has utterly neglected to attend to them.\n\nHe has refused to pass other Laws for the accommodation of large districts of people, unless those people would relinquish the right of Representation in the Legislature, a right inestimable to them and formidable to tyrants only.\n\nHe has called together legislative bodies at places unusual, uncomfortable, and distant from the depository of their public Records, for the sole purpose of fatiguing them into compliance with his measures.\n\nHe has dissolved Representative Houses repeatedly, for opposing with manly firmness his invasions on the rights of the people.\n\nHe has refused for a long time, after such dissolutions, to cause others to be elected; whereby the Legislative powers, incapable of Annihilation, have returned to the People at large for their exercise; the State remaining in the mean time exposed to all the dangers of invasion from without, and convulsions within.\n\nHe has endeavoured to prevent the population of these States; for that purpose obstructing the Laws for Naturalization of Foreigners; refusing to pass others to encourage their migrations hither, and raising the conditions of new Appropriations of Lands.\n\nHe has obstructed the Administration of Justice, by refusing his Assent to Laws for establishing Judiciary powers.\n\nHe has made Judges dependent on his Will alone, for the tenure of their offices, and the amount and payment of their salaries.\n\nHe has erected a multitude of New Offices, and sent hither swarms of Officers to harrass our people, and eat out their substance.\n\nHe has kept among us, in times of peace, Standing Armies without the Consent of our legislatures.\n\nHe has affected to render the Military independent of and superior to the Civil power.\n\nHe has combined with others to subject us to a jurisdiction foreign to our constitution, and unacknowledged by our laws; giving his Assent to their Acts of pretended Legislation:\n\nFor Quartering large bodies of armed troops among us:\n\nFor protecting them, by a mock Trial, from punishment for any Murders which they should commit on the Inhabitants of these States:\n\nFor cutting off our Trade with all parts of the world:\n\nFor imposing Taxes on us without our Consent:\n\nFor depriving us in many cases, of the benefits of Trial by Jury:\n\nFor transporting us beyond Seas to be tried for pretended offences\n\nFor abolishing the free System of English Laws in a neighbouring Province, establishing therein an Arbitrary government, and enlarging its Boundaries so as to render it at once an example and fit instrument for introducing the same absolute rule into these Colonies:\n\nFor taking away our Charters, abolishing our most valuable Laws, and altering fundamentally the Forms of our Governments:\n\nFor suspending our own Legislatures, and declaring themselves invested with power to legislate for us in all cases whatsoever.\n\nHe has abdicated Government here, by declaring us out of his Protection and waging War against us.\n\nHe has plundered our seas, ravaged our Coasts, burnt our towns, and destroyed the lives of our people.\n\nHe is at this time transporting large Armies of foreign Mercenaries to compleat the works of death, desolation and tyranny, already begun with circumstances of Cruelty & perfidy scarcely paralleled in the most barbarous ages, and totally unworthy the Head of a civilized nation.\n\nHe has constrained our fellow Citizens taken Captive on the high Seas to bear Arms against their Country, to become the executioners of their friends and Brethren, or to fall themselves by their Hands.\n\nHe has excited domestic insurrections amongst us, and has endeavoured to bring on the inhabitants of our frontiers, the merciless Indian Savages, whose known rule of warfare, is an undistinguished destruction of all ages, sexes and conditions.\n\nIn every stage of these Oppressions We have Petitioned for Redress in the most humble terms: Our repeated Petitions have been answered only by repeated injury. A Prince whose character is thus marked by every act which may define a Tyrant, is unfit to be the ruler of a free people.\n\nNor have We been wanting in attentions to our Brittish brethren. We have warned them from time to time of attempts by their legislature to extend an unwarrantable jurisdiction over us. We have reminded them of the circumstances of our emigration and settlement here. We have appealed to their native justice and magnanimity, and we have conjured them by the ties of our common kindred to disavow these usurpations, which, would inevitably interrupt our connections and correspondence. They too have been deaf to the voice of justice and of consanguinity. We must, therefore, acquiesce in the necessity, which denounces our Separation, and hold them, as we hold the rest of mankind, Enemies in War, in Peace Friends.\n\nWe, therefore, the Representatives of the united States of America, in General Congress, Assembled, appealing to the Supreme Judge of the world for the rectitude of our intentions, do, in the Name, and by Authority of the good People of these Colonies, solemnly publish and declare, That these United Colonies are, and of Right ought to be Free and Independent States; that they are Absolved from all Allegiance to the British Crown, and that all political connection between them and the State of Great Britain, is and ought to be totally dissolved; and that as Free and Independent States, they have full Power to levy War, conclude Peace, contract Alliances, establish Commerce, and to do all other Acts and Things which Independent States may of right do. And for the support of this Declaration, with a firm reliance on the protection of divine Providence, we mutually pledge to each other our Lives, our Fortunes and our sacred Honor."
#define DECLARATION_SIZE 8148

std::vector<uint8_t> random_vector_uint8_t(uint8_t first, uint8_t last, size_t size) {
    // Random vector
    std::vector<uint8_t> rv(size);

    // Random device
    std::random_device rnd_device;

	// Specify the engine and distribution
    std::mt19937 mersenne_engine {rnd_device()};

	// Uniform ints
	std::uniform_int_distribution<uint8_t> dist {first, last};

	// Create the lamba function
    auto gen = [&dist, &mersenne_engine]() {return dist(mersenne_engine);};

	// Use std generate to fill the random array
    std::generate(std::begin(rv),
		std::end(rv), gen);

    // print_vector(rv);
    return rv;
}

void run_4_kernels(uint8_t * dest, uint8_t * host, uint8_t * rv, size_t allocated_size, size_t block_size) {
    const unsigned int n_threads = allocated_size;
    const unsigned int n_blocks = (n_threads / block_size) > 0 ? n_threads / block_size : 1;
    add_matrices<<<n_blocks, block_size>>>(dest, host, rv);
    subtract_matrices<<<n_blocks, block_size>>>(dest, host, rv);
    multiply_matrices<<<n_blocks, block_size>>>(dest, host, rv);
    modulo_matrices<<<n_blocks, block_size>>>(dest, host, rv);
}

void run_pageable(uint8_t * results, uint8_t * host, uint8_t * rv, size_t allocated_size, size_t block_size) {
    // Allocate GPU Pinned Memory
    uint8_t * device_mem;
    cudaMalloc((void **)&device_mem, allocated_size);
    uint8_t * device_rv;
    cudaMalloc((void **)&device_rv, allocated_size);
    uint8_t * device_results;
    cudaMalloc((void **)&device_results, allocated_size);

    // std::cout << (char * ) host << std::endl;
    // print_vector(std::vector<uint8_t>(host, host+allocated_size));
    // print_vector(std::vector<uint8_t>(rv, rv+allocated_size));

    // Copy host memory to GPU memory
    cudaMemcpy(device_mem, host,
        allocated_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rv, rv,
        allocated_size, cudaMemcpyHostToDevice);

    // Run the 4 functions
    run_4_kernels(device_results, device_mem, device_rv, allocated_size, block_size);

    // Copy results to host memory
    cudaMemcpy(results, device_results,
        allocated_size, cudaMemcpyDeviceToHost);

    // print_vector(std::vector<uint8_t>(results, results+allocated_size));

    // Free the device memory
    cudaFree(device_mem);
    cudaFree(device_rv);
    cudaFree(device_results);
}

void run_pinnable(uint8_t * results, uint8_t * host, uint8_t * rv, size_t allocated_size, size_t block_size) {
    // Allocate GPU Pinned Memory
    uint8_t * device_mem;
    cudaMallocHost((void **)&device_mem, allocated_size);
    uint8_t * device_rv;
    cudaMallocHost((void **)&device_rv, allocated_size);
    uint8_t * device_results;
    cudaMallocHost((void **)&device_results, allocated_size);

    // std::cout << (char * ) host << std::endl;
    // print_vector(std::vector<uint8_t>(host, host+allocated_size));
    // print_vector(std::vector<uint8_t>(rv, rv+allocated_size));

    // Copy host memory to GPU memory
    cudaMemcpy(device_mem, host,
        allocated_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rv, rv,
        allocated_size, cudaMemcpyHostToDevice);

    // Run the 4 functions
    run_4_kernels(device_results, device_mem, device_rv, allocated_size, block_size);

    // Copy results to host memory
    cudaMemcpy(results, device_results,
        allocated_size, cudaMemcpyDeviceToHost);

    // print_vector(std::vector<uint8_t>(results, results+allocated_size));

    // Free the device memory
    cudaFree(device_mem);
    cudaFree(device_rv);
    cudaFree(device_results);
}

void caesar_cipher(uint8_t * results, uint8_t * host, size_t allocated_size, size_t block_size, int offset) {
    const unsigned int n_threads = allocated_size;
    const unsigned int n_blocks = (n_threads / block_size) > 0 ? n_threads / block_size : 1;
    // Allocate GPU Pinned Memory
    uint8_t * device_mem;
    cudaMalloc((void **)&device_mem, allocated_size);
    uint8_t * device_results;
    cudaMalloc((void **)&device_results, allocated_size);

    // Copy host memory to GPU memory
    cudaMemcpy(device_mem, host,
        allocated_size, cudaMemcpyHostToDevice);

    // Run the cipher
    add_constant<<<n_blocks, block_size>>>(device_results, device_mem, offset);

    // Copy results to host memory
    cudaMemcpy(results, device_results,
        allocated_size, cudaMemcpyDeviceToHost);

    // print_vector(std::vector<uint8_t>(results, results+allocated_size));

    // Free the device memory
    cudaFree(device_mem);
    cudaFree(device_results);
}



int main(int argc, char * argv[]) {
    // Parse command line
    if (argc != 3) {
        std::cout << "Usage: ./file [cipher offset] [blockset]" << std::endl;
		return 0;
    };

    // Caesar cipher offset
    long offset = std::stol(std::string(argv[1]));
    long block_size = std::stol(std::string(argv[2]));

    // Allocate the host buffers
    size_t alloc_size = std::pow(2, std::ceil(std::log2(DECLARATION_SIZE)));
    // size_t alloc_size = 1<<27;
    uint8_t * host_pageable_mem = (uint8_t *) std::malloc(alloc_size);
    std::strncpy((char *)host_pageable_mem, DECLARATION, DECLARATION_SIZE);
    std::cout << "Alloc size " << alloc_size << std::endl;

    // Allocate another random vector to do operations with
    std::vector<uint8_t> rv = random_vector_uint8_t(1, 10, alloc_size);
    // std::cout << (char * ) host_pageable_mem << std::endl;
    // print_vector(rv);

    // Allocate a destination buffer
    uint8_t * dest = (uint8_t *) std::malloc(alloc_size);

    // Run pageable
    TIC();
    run_pageable(dest, host_pageable_mem, rv.data(), alloc_size, block_size);
    std::cout << "Pageable took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    // print_vector(std::vector<uint8_t>(dest, dest+alloc_size));

    // Run pinnable
    TIC();
    run_pinnable(dest, host_pageable_mem, rv.data(), alloc_size, block_size);
    std::cout << "Pinnable took " << TOC<std::chrono::microseconds>() << " microseconds" << std::endl;
    // print_vector(std::vector<uint8_t>(dest, dest+alloc_size));


    // Run caesar cipher
    std::string declaration(DECLARATION);
    caesar_cipher((uint8_t *)declaration.data(), host_pageable_mem, declaration.size(), block_size, 1);
    // print_vector(std::vector<uint8_t> (declaration.data(), declaration.data() + DECLARATION_SIZE));
    std::cout << "CIPHER TEXT:\n" << std::string(declaration.begin(), declaration.begin() + DECLARATION_SIZE) << std::endl;

    caesar_cipher((uint8_t *)declaration.data(), (uint8_t *)declaration.data(), declaration.size(), block_size, -1);
    // print_vector(std::vector<uint8_t> (declaration.data(), declaration.data() + DECLARATION_SIZE));
    std::cout << "DECODED TEXT:\n" << std::string(declaration.begin(), declaration.begin() + DECLARATION_SIZE) << std::endl;


    // Free memory
    std::free(host_pageable_mem);
    std::free(dest);

    return 0;
}